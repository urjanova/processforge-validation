from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr
import boto3

class ProcessForgeValidator:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def _load_dataframe_from_zarr(self,store_path):
        store = zarr.storage.LocalStore(store_path)
        root = zarr.open(store=store, mode="r")
        streams = sorted(root.group_keys())
        rows = []
        components = set()
        mode = root.attrs.get("mode", "steady")
        for stream in streams:
            group = root[stream]
            comp_group = group.get("__composition__")
            comp_names = sorted(comp_group.keys()) if comp_group is not None else []
            components.update(comp_names)
            has_time = "time" in group and mode == "dynamic"
            length = group["time"].shape[0] if has_time else 1
            for idx in range(length):
                rows.append(_build_dataframe_row(group, stream, idx, comp_names, has_time))
        df = pd.DataFrame(rows)
        for comp in sorted(components):
            if comp in df:
                df[comp] = df[comp].fillna(0.0)
            else:
                df[comp] = 0.0
        return df
    def generate_validation_excel(self, data_source, output_filename):
        """
        Generate a multi-sheet Excel validation report from simulation results.

        data_source: path to a Zarr store, CSV file, or pandas DataFrame.
        output_filename: path for the output .xlsx file.
        """
        if isinstance(data_source, str) and os.path.isdir(data_source):
            df = _load_dataframe_from_zarr(data_source)
        elif isinstance(data_source, str):
            df = pd.read_csv(data_source)
        else:
            df = data_source.copy()

        if "Stream" in df.columns and "stream" not in df.columns:
            df.rename(columns={"Stream": "stream"}, inplace=True)

        known_cols = {
            "time",
            "stream",
            "T [K]",
            "P [Pa]",
            "Phase",
            "VaporFrac",
            "flowrate",
        }
        comp_cols = [c for c in df.columns if c not in known_cols]
        numeric_comp = (
            df[comp_cols].apply(pd.to_numeric, errors="coerce")
            if comp_cols
            else pd.DataFrame()
        )
        if not numeric_comp.empty:
            df["Total_Fraction"] = numeric_comp.sum(axis=1)
            df["Composition_Alert"] = np.where(
                np.isclose(df["Total_Fraction"], 1.0, atol=1e-5),
                "OK",
                "MASS LEAK",
            )
        else:
            df["Total_Fraction"] = ""
            df["Composition_Alert"] = ""
        mass_ok = (df["Composition_Alert"] == "OK").all()

        pump_check = pd.DataFrame()
        pump_ok = True
        temp_ok = True

        if "stream" in df.columns and "time" in df.columns:
            stream_names = df["stream"].unique()
            pump_ins = sorted([s for s in stream_names if "before_pump" in str(s)])
            pump_outs = sorted([s for s in stream_names if "after_pump" in str(s)])

            for p_in, p_out in zip(pump_ins, pump_outs):
                df_in = df[df["stream"] == p_in].set_index("time")
                df_out = df[df["stream"] == p_out].set_index("time")
                common_idx = df_in.index.intersection(df_out.index)
                if common_idx.empty:
                    continue
                pc = pd.DataFrame(index=common_idx)
                pc["Pump"] = f"{p_in} -> {p_out}"
                pc["Pressure_Gain_Pa"] = (
                    df_out.loc[common_idx, "P [Pa]"].values
                    - df_in.loc[common_idx, "P [Pa]"].values
                )
                pc["Temp_Rise_K"] = (
                    df_out.loc[common_idx, "T [K]"].values
                    - df_in.loc[common_idx, "T [K]"].values
                )
                pc["Pump_Status"] = np.where(
                    pc["Pressure_Gain_Pa"] > 0,
                    "Functional",
                    "Broken",
                )
                pump_check = pd.concat([pump_check, pc])

            if not pump_check.empty:
                pump_ok = (pump_check["Pump_Status"] == "Functional").all()
                temp_ok = (pump_check["Temp_Rise_K"] >= 0).all()

        summary_rows = [
            {
                "Physical Law": "Conservation of Mass",
                "Logic": "Do chemical fractions add to 1.0?",
                "Status": "PASS" if mass_ok else "FAIL",
            }
        ]
        if not pump_check.empty:
            summary_rows.append(
                {
                    "Physical Law": "Pump Work (Pressure)",
                    "Logic": "Does the pump increase pressure?",
                    "Status": "PASS" if pump_ok else "FAIL",
                }
            )
            summary_rows.append(
                {
                    "Physical Law": "Thermal Direction",
                    "Logic": "Is the outlet temperature >= inlet?",
                    "Status": "PASS" if temp_ok else "WARNING",
                }
            )
        summary_df = pd.DataFrame(summary_rows)

        _ensure_parent_dir(output_filename)
        with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="1_EXECUTIVE_SUMMARY", index=False)
            if not pump_check.empty:
                pump_check.to_excel(writer, sheet_name="2_PUMP_PERFORMANCE")
            df.to_excel(writer, sheet_name="3_RAW_DATA_CHECKED", index=False)

        logger.info(f"Validation Report Generated: {output_filename}")
