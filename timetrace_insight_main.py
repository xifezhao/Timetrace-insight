import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# --- 0. Load Data ---
data_dir = "" # Assume CSVs are in the same directory as the script, or specify path
csv_files = ["clinical_data.csv", "lab_data.csv", "track_names.csv", "clinical_parameters.csv", "lab_parameters.csv"]
dfs = {}
for file_name in csv_files:
    file_path = os.path.join(data_dir, file_name)
    print(f"\n===== Loading {file_name} =====")
    try:
        dfs[file_name] = pd.read_csv(file_path)
        print("行数 × 列数:", dfs[file_name].shape)
    except Exception as e:
        print(f"无法读取 {file_name}: {e}")
        dfs[file_name] = None

clinical_df = dfs.get("clinical_data.csv")
lab_df = dfs.get("lab_data.csv")
# lab_params_df = dfs.get("lab_parameters.csv") # Not directly used
# clinical_params_df = dfs.get("clinical_parameters.csv") # Not directly used

# --- Helper: Select a case for detailed demonstration (for non-batch parts) ---
CASE_ID_TO_DEMO = 1 # Still used for some initial log prints
if clinical_df is not None:
    print(f"\n--- Focusing on CASE_ID: {CASE_ID_TO_DEMO} for some detailed logs ---")

# --- 1. Temporal-Centric Design (f_align) ---
print("\n--- 1. Temporal-Centric Design (f_align) ---")
if lab_df is not None:
    lab_df_aligned = lab_df.sort_values(by=['caseid', 'dt'])
    print("Lab data is sorted by caseid and dt (D_aligned).")
else:
    lab_df_aligned = pd.DataFrame()
    print("Lab data not available for alignment.")

# --- Mock Knowledge Base (K) ---
mock_K = {
    'lab_thresholds': {
        'hb': {'name': 'Hemoglobin', 'unit': 'g/dL', 'low': 10, 'normal_low': 12, 'normal_high': 16, 'high': 18, 'critical_low': 7, 'critical_high': 20},
        'gluc': {'name': 'Glucose', 'unit': 'mg/dL', 'low': 60, 'normal_low': 70, 'normal_high': 110, 'high': 125, 'critical_low': 40, 'critical_high': 400}
    },
    'rules': [
        {'id': 'R001', 'name': 'Preoperative Anemia Female', 'condition_text': "(row['sex'] == 0) and ('preop_hb' in row) and pd.notna(row['preop_hb']) and (row['preop_hb'] < 12)", 'annotation': 'PreOp Anemia (F)', 'source': 'K_expert_simple', 'type': 'clinical_summary'},
        {'id': 'R002', 'name': 'Preoperative Anemia Male', 'condition_text': "(row['sex'] == 1) and ('preop_hb' in row) and pd.notna(row['preop_hb']) and (row['preop_hb'] < 13)", 'annotation': 'PreOp Anemia (M)', 'source': 'K_expert_simple', 'type': 'clinical_summary'},
        {'id': 'R003', 'name': 'Hypoglycemia Alert', 'condition_text': "(lab_name == 'gluc') and pd.notna(lab_result) and (lab_result < 60)", 'annotation': 'Hypoglycemia Alert!', 'source': 'K_expert_lab', 'type': 'lab_point'},
        {'id': 'R004', 'name': 'Critical Low Hemoglobin', 'condition_text': "(lab_name == 'hb') and pd.notna(lab_result) and (lab_result < mock_K['lab_thresholds']['hb']['critical_low'])", 'annotation': 'Critical Low HB!', 'source': 'K_CPG_like', 'type': 'lab_point'},
        {'id': 'R005', 'name': 'Hyperglycemia Concern', 'condition_text': "(lab_name == 'gluc') and pd.notna(lab_result) and (lab_result > 180)", 'annotation': 'Hyperglycemia Concern', 'source': 'K_expert_lab', 'type': 'lab_point'},
        {'id': 'R006', 'name': 'Sustained High Glucose (State)', 'condition_text': "('gluc_state' in locals()) and (gluc_state == 'High GLUC')", 'annotation': 'Sustained High GLUC State', 'source': 'K_derived_state', 'type': 'lab_state_point'},
        {'id': 'R007', 'name': 'PostOp Risk Window', 'condition_text': "True", 'annotation': 'PostOp Risk Window (1-3h)', 'source': 'K_guideline_sim', 'type': 'interval', 'interval_logic': lambda opstart_val: (opstart_val + 3600, opstart_val + 3*3600) if pd.notna(opstart_val) else (None, None)}
    ]
}
print("\n--- Mock Knowledge Base (K) Initialized ---")

# --- 2. Data Preprocessing & Feature Engineering Layer (L_process) ---
print("\n--- 2. Data Preprocessing & Feature Engineering Layer (L_process) ---")
# --- 2.1. Data Cleaning (f_clean) ---
print("\n--- 2.1. Data Cleaning (f_clean) ---")
def f_clean(df_aligned, lab_name_to_clean='hb', outlier_method='iqr', iqr_multiplier=1.5):
    if df_aligned is None or df_aligned.empty:
        print(f"Input DataFrame for f_clean is None or empty for {lab_name_to_clean}. Skipping.")
        return df_aligned
    df_cleaned = df_aligned.copy()
    param_df_for_outlier = df_cleaned[df_cleaned['name'] == lab_name_to_clean].copy()
    if not param_df_for_outlier.empty:
        if outlier_method == 'iqr':
            Q1 = param_df_for_outlier['result'].quantile(0.25)
            Q3 = param_df_for_outlier['result'].quantile(0.75)
            IQR = Q3 - Q1
            if pd.notna(IQR) and IQR > 1e-6: # IQR is valid and not essentially zero
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                mask_outlier_indices = param_df_for_outlier[(param_df_for_outlier['result'] < lower_bound) | (param_df_for_outlier['result'] > upper_bound)].index
                num_outliers = len(mask_outlier_indices)
                if num_outliers > 0:
                    print(f"IQR Outlier Detection for '{lab_name_to_clean}': Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, Lower={lower_bound:.2f}, Upper={upper_bound:.2f}. {num_outliers} outliers marked as NaN.")
                    df_cleaned.loc[mask_outlier_indices, 'result'] = np.nan
            else:
                print(f"IQR is zero, NaN, or too small for '{lab_name_to_clean}', skipping IQR outlier detection based on it.")
        
        lab_param_mask_in_df_cleaned = (df_cleaned['name'] == lab_name_to_clean)
        temp_impute_df = df_cleaned[lab_param_mask_in_df_cleaned].sort_values(by=['caseid', 'dt'])
        
        # 确保 'result' 是数值型
        temp_impute_df['result'] = pd.to_numeric(temp_impute_df['result'], errors='coerce')
        
        # 对包含 'caseid' 和 'result' 的 DataFrame 进行 groupby，然后对 'result' 列应用 ffill
        # groupby 返回一个 DataFrameGroupBy 对象，从中选择 'result' 列，然后应用 ffill
        temp_impute_df['result_ffilled'] = temp_impute_df.groupby('caseid')['result'].ffill() # 将填充结果赋给新列
        
        # 将填充后的值更新回 df_cleaned
        # 使用 temp_impute_df 的索引来定位 df_cleaned 中的相应行
        df_cleaned.loc[temp_impute_df.index, 'result'] = temp_impute_df['result_ffilled']
        
        print(f"Imputation (ffill) applied for '{lab_name_to_clean}'.")
    else: print(f"No data for '{lab_name_to_clean}' to perform outlier detection.")
    return df_cleaned

# --- 2.2. Feature Derivation (f_derive) ---
print("\n--- 2.2. Feature Derivation (f_derive) ---")
if not D_clean_clinical.empty:
    if 'height' in D_clean_clinical.columns and 'weight' in D_clean_clinical.columns:
        D_clean_clinical['bmi_derived'] = D_clean_clinical['weight'] / ((D_clean_clinical['height'] / 100) ** 2)
        if CASE_ID_TO_DEMO in D_clean_clinical['caseid'].unique():
            case_demo_bmi_series = D_clean_clinical[D_clean_clinical['caseid'] == CASE_ID_TO_DEMO]['bmi_derived']
            if not case_demo_bmi_series.empty: print(f"Derived BMI for CASE_ID {CASE_ID_TO_DEMO}: {case_demo_bmi_series.iloc[0]:.2f}")
if D_clean is not None and not D_clean.empty:
    for lab_to_delta in ['hb', 'gluc']:
        temp_delta_calc_df = D_clean[D_clean['name'] == lab_to_delta][['caseid', 'dt', 'result']].copy()
        if not temp_delta_calc_df.empty:
            temp_delta_calc_df = temp_delta_calc_df.sort_values(by=['caseid', 'dt'])
            temp_delta_calc_df[f'delta_{lab_to_delta}'] = temp_delta_calc_df.groupby('caseid')['result'].diff()
            temp_delta_calc_df['name'] = lab_to_delta
            if f'delta_{lab_to_delta}' not in D_clean.columns: D_clean[f'delta_{lab_to_delta}'] = np.nan
            D_clean = pd.merge(D_clean, temp_delta_calc_df[['caseid', 'dt', 'name', f'delta_{lab_to_delta}']], on=['caseid', 'dt', 'name'], how='left', suffixes=('_orig', ''))
            if f'delta_{lab_to_delta}_orig' in D_clean.columns: D_clean[f'delta_{lab_to_delta}'] = D_clean[f'delta_{lab_to_delta}'].fillna(D_clean[f'delta_{lab_to_delta}_orig']); D_clean.drop(columns=[f'delta_{lab_to_delta}_orig'], inplace=True)
    print(f"Derived delta values for relevant labs and added/updated in D_clean.")
else: print("D_clean is None or empty, skipping delta derivation.")

# --- 2.3. Algorithm 1: Temporal Feature Abstraction (TFA - f_abstract) ---
print("\n--- 2.3. Algorithm 1: Temporal Feature Abstraction (TFA - f_abstract) ---")
F_abstract_S, F_abstract_T, F_abstract_P, F_abstract_R = {}, {}, {}, {}
SHAPE_TEMPLATES = { "rapid_rise_plateau": np.array([0,0.3,0.8,1,1,0.95]), "v_shape_dip": np.array([1,0.5,0.1,0.2,0.6,1]), "gradual_decline": np.array([1,0.9,0.75,0.6,0.4,0.2]) }
SHAPE_WINDOW_SIZE = 6
def match_shape_in_series(series_values, shape_template_values, tolerance=0.2):
    if len(series_values) != len(shape_template_values): return False
    if np.isnan(series_values).any() or np.isnan(shape_template_values).any(): return False
    min_s, max_s = np.min(series_values), np.max(series_values); min_t, max_t = np.min(shape_template_values), np.max(shape_template_values)
    if abs(max_s - min_s) < 1e-9 or abs(max_t - min_t) < 1e-9 : return False
    series_norm = (series_values - min_s) / (max_s - min_s); template_norm = (shape_template_values - min_t) / (max_t - min_t)
    mse = np.mean((series_norm - template_norm)**2); return mse < tolerance

def f_abstract(d_clean_lab, d_clean_clinical, knowledge_base, enable_shape_abstraction=True, min_support_count_gsp=5):
    global F_abstract_S, F_abstract_T, F_abstract_P, F_abstract_R
    if d_clean_lab is None or d_clean_lab.empty: print("f_abstract: d_clean_lab is None or empty."); return pd.DataFrame()
    processed_lab_data = d_clean_lab.copy()
    for lab_for_state in ['hb', 'gluc']:
        if lab_for_state in knowledge_base['lab_thresholds']:
            thresholds = knowledge_base['lab_thresholds'][lab_for_state]; state_col_name = f'{lab_for_state}_state'
            if state_col_name not in processed_lab_data.columns: processed_lab_data[state_col_name] = 'N/A'
            mask_current_lab = (processed_lab_data['name'] == lab_for_state)
            if 'result' in processed_lab_data.columns and pd.api.types.is_numeric_dtype(processed_lab_data['result']):
                current_lab_results = processed_lab_data.loc[mask_current_lab, 'result']
                if not current_lab_results.empty:
                    conditions = [(current_lab_results < thresholds['critical_low']), (current_lab_results < thresholds['low']), (current_lab_results < thresholds['normal_low']), (current_lab_results <= thresholds['normal_high']), (current_lab_results < thresholds['high']), (current_lab_results >= thresholds['high'])]
                    states_values = [f'Critical Low {lab_for_state.upper()}', f'Low {lab_for_state.upper()}', f'Mildly Low {lab_for_state.upper()}', f'Normal {lab_for_state.upper()}', f'Mildly High {lab_for_state.upper()}', f'High {lab_for_state.upper()}']
                    processed_lab_data.loc[mask_current_lab, state_col_name] = np.select(conditions, states_values, default='N/A')
            F_abstract_S[lab_for_state] = processed_lab_data[mask_current_lab][['caseid', 'dt', state_col_name]]
            if CASE_ID_TO_DEMO in processed_lab_data['caseid'].unique(): print(f"State Identification (S) for '{lab_for_state}' (CASE_ID {CASE_ID_TO_DEMO} sample available).")
    for lab_for_trend in ['hb', 'gluc']:
        delta_col = f'delta_{lab_for_trend}'
        if delta_col in processed_lab_data.columns:
            trend_col_name = f'{lab_for_trend}_trend'
            if trend_col_name not in processed_lab_data.columns: processed_lab_data[trend_col_name] = 'N/A'
            mask_current_lab_trend = (processed_lab_data['name'] == lab_for_trend)
            current_lab_deltas = processed_lab_data.loc[mask_current_lab_trend, delta_col]
            if not current_lab_deltas.empty and pd.api.types.is_numeric_dtype(current_lab_deltas):
                trend_threshold = 0.2 if lab_for_trend == 'hb' else 5
                conditions_trend = [(current_lab_deltas > trend_threshold), (current_lab_deltas < -trend_threshold), (current_lab_deltas.notna())]
                trends_values = ['Increasing', 'Decreasing', 'Stable']
                processed_lab_data.loc[mask_current_lab_trend, trend_col_name] = np.select(conditions_trend, trends_values, default='N/A')
            F_abstract_T[lab_for_trend] = processed_lab_data[mask_current_lab_trend][['caseid', 'dt', trend_col_name]]
            if CASE_ID_TO_DEMO in processed_lab_data['caseid'].unique(): print(f"Trend Analysis (T) for '{lab_for_trend}' (CASE_ID {CASE_ID_TO_DEMO} sample available).")
    if enable_shape_abstraction:
        print("\nShape-based Feature Abstraction:")
        for lab_for_shape in ['hb', 'gluc']:
            shape_col_name = f'{lab_for_shape}_shape'; F_abstract_S[f'{lab_for_shape}_shapes'] = []
            if shape_col_name not in processed_lab_data.columns: processed_lab_data[shape_col_name] = 'No_Shape'
            unique_case_ids_for_shape = processed_lab_data[processed_lab_data['name'] == lab_for_shape]['caseid'].unique()
            for caseid_val in unique_case_ids_for_shape:
                case_lab_data_shape = processed_lab_data[(processed_lab_data['caseid'] == caseid_val) & (processed_lab_data['name'] == lab_for_shape)].sort_values('dt')
                if len(case_lab_data_shape) >= SHAPE_WINDOW_SIZE:
                    for i in range(len(case_lab_data_shape) - SHAPE_WINDOW_SIZE + 1):
                        window_indices = case_lab_data_shape.index[i : i + SHAPE_WINDOW_SIZE]; window_values = case_lab_data_shape.loc[window_indices, 'result'].values
                        if np.isnan(window_values).any(): continue
                        for shape_name, template_vals in SHAPE_TEMPLATES.items():
                            if match_shape_in_series(window_values, template_vals):
                                start_dt = case_lab_data_shape.loc[window_indices[0], 'dt']; processed_lab_data.loc[window_indices[0], shape_col_name] = shape_name
                                F_abstract_S[f'{lab_for_shape}_shapes'].append({'caseid': caseid_val, 'dt_start': start_dt, 'shape_name': shape_name, 'lab': lab_for_shape})
                                if caseid_val == CASE_ID_TO_DEMO: print(f"  Shape '{shape_name}' matched for '{lab_for_shape}' in CASE_ID {caseid_val} at dt={start_dt:.0f}")
            if CASE_ID_TO_DEMO in processed_lab_data['caseid'].unique():
                 if not F_abstract_S[f'{lab_for_shape}_shapes'] or not any(s['caseid'] == CASE_ID_TO_DEMO for s in F_abstract_S[f'{lab_for_shape}_shapes']): print(f"  No predefined shapes matched for '{lab_for_shape}' in CASE_ID {CASE_ID_TO_DEMO}.")
    print("\nEnhanced Sequential Pattern Mining (GSP-like):")
    all_events_gsp = []; F_abstract_P['frequent_event_pairs'] = {}; F_abstract_P['frequent_event_triplets'] = {}
    for _, row_gsp in processed_lab_data.iterrows():
        for lab_param_gsp in ['hb', 'gluc']:
            state_col_gsp = f'{lab_param_gsp}_state'
            if state_col_gsp in processed_lab_data.columns and pd.notna(row_gsp[state_col_gsp]) and row_gsp[state_col_gsp] != 'N/A' and row_gsp['name'] == lab_param_gsp:
                all_events_gsp.append({'caseid': row_gsp['caseid'], 'dt': row_gsp['dt'], 'event_label': f"{lab_param_gsp}:{row_gsp[state_col_gsp]}"})
    event_df_gsp = pd.DataFrame(all_events_gsp)
    if not event_df_gsp.empty:
        event_df_gsp = event_df_gsp.drop_duplicates().sort_values(by=['caseid', 'dt'])
        frequent_single_events_gsp = event_df_gsp['event_label'].value_counts(); frequent_single_events_gsp = frequent_single_events_gsp[frequent_single_events_gsp >= min_support_count_gsp].index.tolist()
        print(f"  Frequent single events (L1, min_support={min_support_count_gsp}): {frequent_single_events_gsp if frequent_single_events_gsp else 'None'}")
        candidate_pairs_gsp = {}
        for _, group_gsp in event_df_gsp.groupby('caseid'):
            sequence_gsp = group_gsp['event_label'].tolist()
            for i in range(len(sequence_gsp) - 1):
                if sequence_gsp[i] in frequent_single_events_gsp and sequence_gsp[i+1] in frequent_single_events_gsp: candidate_pairs_gsp[(sequence_gsp[i], sequence_gsp[i+1])] = candidate_pairs_gsp.get((sequence_gsp[i], sequence_gsp[i+1]), 0) + 1
        F_abstract_P['frequent_event_pairs'] = {p: c for p, c in candidate_pairs_gsp.items() if c >= min_support_count_gsp}
        print(f"  Frequent event pairs (L2): {F_abstract_P['frequent_event_pairs'] if F_abstract_P['frequent_event_pairs'] else 'None'}")
        candidate_triplets_gsp = {}
        if F_abstract_P['frequent_event_pairs']:
            for _, group_gsp_trip in event_df_gsp.groupby('caseid'):
                sequence_gsp_trip = group_gsp_trip['event_label'].tolist()
                for i in range(len(sequence_gsp_trip) - 2):
                    if (sequence_gsp_trip[i], sequence_gsp_trip[i+1]) in F_abstract_P['frequent_event_pairs'] and (sequence_gsp_trip[i+1], sequence_gsp_trip[i+2]) in F_abstract_P['frequent_event_pairs']: candidate_triplets_gsp[(sequence_gsp_trip[i], sequence_gsp_trip[i+1], sequence_gsp_trip[i+2])] = candidate_triplets_gsp.get((sequence_gsp_trip[i], sequence_gsp_trip[i+1], sequence_gsp_trip[i+2]), 0) + 1
            F_abstract_P['frequent_event_triplets'] = {t: c for t, c in candidate_triplets_gsp.items() if c >= min_support_count_gsp}
            print(f"  Frequent event triplets (L3): {F_abstract_P['frequent_event_triplets'] if F_abstract_P['frequent_event_triplets'] else 'None'}")
    else: print("  No events for GSP-like pattern mining.")
    F_abstract_R['info'] = "Representation Learning - Placeholder"
    return processed_lab_data

D_clean_lab_abstracted = f_abstract(D_clean, D_clean_clinical, mock_K, enable_shape_abstraction=True, min_support_count_gsp=10)

# --- 3. Algorithm 2: Intelligent Data-Knowledge Linking (Φ) ---
print("\n--- 3. Algorithm 2: Intelligent Data-Knowledge Linking (Φ) ---")
def intelligent_data_knowledge_linking(d_clean_lab_abstracted_input, d_clean_clinical_input, knowledge_base):
    current_annotations_link = []
    df_clinical_link = d_clean_clinical_input.copy() if d_clean_clinical_input is not None else pd.DataFrame()
    df_lab_link = d_clean_lab_abstracted_input.copy() if d_clean_lab_abstracted_input is not None else pd.DataFrame()
    # print("Semantic Mapping (m): Implicitly handled.")
    # print("Rule-Based Inference (r): Applying rules from Mock K...")
    for rule_link in knowledge_base['rules']:
        rule_type_link = rule_link.get('type', 'unknown'); annotation_details_link = {'annotation': rule_link['annotation'], 'rule_id': rule_link['id'], 'source': rule_link['source'], 'type': rule_type_link}
        if rule_type_link == 'clinical_summary':
            if not df_clinical_link.empty:
                for _, row_clin_link in df_clinical_link.iterrows():
                    try:
                        if 'preop_hb' in row_clin_link or 'preop_hb' not in rule_link['condition_text']:
                             if eval(rule_link['condition_text'], {"row": row_clin_link, "mock_K": knowledge_base}):
                                ann_link = {'caseid': row_clin_link['caseid'], 'dt': None, 'dt_start': None, 'dt_end': None, **annotation_details_link}; current_annotations_link.append(ann_link)
                                # if row_clin_link['caseid'] == CASE_ID_TO_DEMO: print(f"  Clinical Summary Rule '{rule_link['id']}' triggered for CASE_ID {row_clin_link['caseid']}")
                    except Exception: pass
        elif rule_type_link in ['lab_point', 'lab_state_point']:
            if not df_lab_link.empty:
                 for _, row_lab_link in df_lab_link.iterrows():
                    try:
                        if pd.notna(row_lab_link.get('result')) or rule_type_link == 'lab_state_point':
                            local_scope_link = {"lab_name": row_lab_link['name'], "lab_result": row_lab_link.get('result'), "mock_K": knowledge_base}
                            for col_link in ['hb_state', 'gluc_state', 'hb_shape', 'gluc_shape']:
                                if col_link in df_lab_link.columns and col_link in row_lab_link and pd.notna(row_lab_link[col_link]): local_scope_link[col_link] = row_lab_link[col_link]
                            if eval(rule_link['condition_text'], {"__builtins__": {}}, local_scope_link):
                                ann_link = {'caseid': row_lab_link['caseid'], 'dt': row_lab_link['dt'], 'dt_start': None, 'dt_end': None, 'param_name': row_lab_link['name'], 'param_value': row_lab_link.get('result'), **annotation_details_link}; current_annotations_link.append(ann_link)
                                # if row_lab_link['caseid'] == CASE_ID_TO_DEMO and pd.notna(row_lab_link['dt']): print(f"  Lab Point/State Rule '{rule_link['id']}' triggered for CASE_ID {row_lab_link['caseid']}")
                    except Exception: pass
        elif rule_type_link == 'interval':
            if not df_clinical_link.empty:
                for _, row_clin_link in df_clinical_link.iterrows():
                    opstart_val = row_clin_link.get('opstart')
                    if 'interval_logic' in rule_link and callable(rule_link['interval_logic']):
                        dt_s, dt_e = rule_link['interval_logic'](opstart_val)
                        if pd.notna(dt_s) and pd.notna(dt_e):
                            ann_link = {'caseid': row_clin_link['caseid'], 'dt': None, 'dt_start': dt_s, 'dt_end': dt_e, **annotation_details_link}; current_annotations_link.append(ann_link)
                            # if row_clin_link['caseid'] == CASE_ID_TO_DEMO: print(f"  Interval Rule '{rule_link['id']}' triggered for CASE_ID {row_clin_link['caseid']}")
    return df_clinical_link, df_lab_link, pd.DataFrame(current_annotations_link)

D_prime_clinical, D_prime_lab, Aknowledge_df = intelligent_data_knowledge_linking(D_clean_lab_abstracted, D_clean_clinical, mock_K)

# --- 4. Algorithm 3: Adaptive Visualization Layout & Rendering (f_layout) ---
if CASE_ID_TO_DEMO is not None:
    print("\n--- 4. Algorithm 3: Adaptive Visualization Layout & Rendering (f_layout) ---")
    def f_layout_logic(d_prime_lab_data_input, case_id_input, view_range_start_dt, view_range_end_dt, theta_density=5):
        if d_prime_lab_data_input is None or d_prime_lab_data_input.empty: print(f"No lab data for layout for case {case_id_input}"); return "No Data"
        # print(f"Simulating layout for CASE_ID {case_id_input}, view range: [{view_range_start_dt}, {view_range_end_dt}], density_threshold: {theta_density}") # Less verbose
        case_data_in_view = d_prime_lab_data_input[ (d_prime_lab_data_input['caseid'] == case_id_input) & (d_prime_lab_data_input['dt'] >= view_range_start_dt) & (d_prime_lab_data_input['dt'] <= view_range_end_dt) ]
        if case_data_in_view.empty: 
            if case_id_input == CASE_ID_TO_DEMO: print(f"  No lab events in view for case {case_id_input}"); 
            return "No Events in View"
        num_events_in_view = case_data_in_view.shape[0]
        if case_id_input == CASE_ID_TO_DEMO: print(f"  Number of lab events in view for case {case_id_input}: {num_events_in_view}")
        view_duration = view_range_end_dt - view_range_start_dt
        if view_duration > 21600: 
            if case_id_input == CASE_ID_TO_DEMO: print("  T_scale: View duration suggests a compressed timescale.")
        else: 
            if case_id_input == CASE_ID_TO_DEMO: print("  T_scale: View duration suggests a linear timescale.")
        if num_events_in_view > theta_density: 
            if case_id_input == CASE_ID_TO_DEMO: print(f"  f_aggregate: Event density ({num_events_in_view}) > threshold ({theta_density}). Consider aggregation."); 
            L_elements = "Aggregated View"
        else: 
            if case_id_input == CASE_ID_TO_DEMO: print(f"  f_aggregate: Event density ({num_events_in_view}) <= threshold ({theta_density}). Display individual events."); 
            L_elements = "Individual Event View"
        if case_id_input == CASE_ID_TO_DEMO: print(f"  L_elements (conceptual output for case {case_id_input}): {L_elements}"); 
        return L_elements

    if D_prime_lab is not None and not D_prime_lab.empty :
        demo_case_lab_data_layout = D_prime_lab[D_prime_lab['caseid'] == CASE_ID_TO_DEMO]
        if not demo_case_lab_data_layout.empty and 'dt' in demo_case_lab_data_layout.columns and demo_case_lab_data_layout['dt'].notna().any():
            min_dt_layout = demo_case_lab_data_layout['dt'].min(); max_dt_layout = demo_case_lab_data_layout['dt'].max()
            if pd.notna(min_dt_layout) and pd.notna(max_dt_layout):
                print("\nLayout for full time range (likely needs aggregation):"); f_layout_logic(D_prime_lab, CASE_ID_TO_DEMO, min_dt_layout, max_dt_layout, theta_density=10)
                print("\nLayout for a shorter time range (e.g., first hour of lab data):"); f_layout_logic(D_prime_lab, CASE_ID_TO_DEMO, min_dt_layout, min_dt_layout + 3600, theta_density=5)
            else: print(f"Min/max dt is NaN for CASE_ID {CASE_ID_TO_DEMO} (layout), skipping.")
        else: print(f"No valid lab data/dt for CASE_ID {CASE_ID_TO_DEMO} (layout).")
    else: print("D_prime_lab not available for layout demonstration.")


# --- 5. Visualizations - Multi-panel per Case ---
print("\n--- 5. Visualizations - Multi-panel per Case ---")
output_viz_dir_multipanel = "all_cases_visualizations_pdf"
if not os.path.exists(output_viz_dir_multipanel): os.makedirs(output_viz_dir_multipanel)
print(f"Multi-panel PDF visualizations will be saved in '{output_viz_dir_multipanel}'.")

SPECIFIC_CASE_IDS_TO_PLOT = [4, 5, 10]

def find_contiguous_time_periods(sorted_times_series, max_gap_seconds=3600*4):
    if sorted_times_series.empty or not pd.api.types.is_numeric_dtype(sorted_times_series): return []
    sorted_times_list = sorted(sorted_times_series.dropna().unique().tolist()) # Corrected line
    if not sorted_times_list: return []
    periods = []; current_start = sorted_times_list[0]; current_end = sorted_times_list[0]
    for t in sorted_times_list[1:]:
        if t <= current_end + max_gap_seconds: current_end = t
        else: periods.append((current_start, current_end)); current_start = t; current_end = t
    periods.append((current_start, current_end)); return periods

def plot_case_summary_figure(case_id, clinical_data, lab_data_abstracted, aknowledge_data, knowledge_base, f_abstract_s_shapes, save_to_dir):
    print(f"  Generating summary figure for Case ID: {case_id}")
    fig, axs = plt.subplots(3, 1, figsize=(15, 20)); fig.suptitle(f'Comprehensive View for Case ID: {case_id}', fontsize=16)
    ax_tl, ax_hb, ax_gluc = axs[0], axs[1], axs[2]
    ax_tl.set_title('Enhanced Timeline'); ax_tl.set_xlabel("Time (s)"); ax_tl.set_yticks([])
    case_clin_tl = clinical_data[clinical_data['caseid'] == case_id]; general_annotations_text = []
    if not case_clin_tl.empty:
        ev_tl = []
        for col_start, col_end, y_pos, color, label_prefix in [('anestart','aneend',0,'skyblue','Anesthesia'), ('opstart','opend',0.05,'lightcoral','Operation')]:
            if col_start in case_clin_tl.columns and col_end in case_clin_tl.columns:
                s, e = case_clin_tl[col_start].iloc[0], case_clin_tl[col_end].iloc[0]
                if pd.notna(s) and pd.notna(e): ax_tl.plot([s,e],[y_pos,y_pos],lw=10,color=color,label=f'{label_prefix} ({s:.0f}s to {e:.0f}s)'); ev_tl.extend([s,e])
        if aknowledge_data is not None and not aknowledge_data.empty:
            case_akn_tl = aknowledge_data[aknowledge_data['caseid']==case_id].copy(); case_akn_tl.sort_values(by=['type', 'dt'], inplace=True); plotted_ann_labels_tl = set()
            for _, ann_r_tl in case_akn_tl.iterrows():
                ann_type = ann_r_tl.get('type', 'unknown'); ann_label_for_legend = f"{ann_r_tl['annotation']} ({ann_r_tl['rule_id']})"
                label_to_use_in_plot = ann_label_for_legend if ann_label_for_legend not in plotted_ann_labels_tl else None
                if label_to_use_in_plot: plotted_ann_labels_tl.add(ann_label_for_legend)
                if ann_type in ['lab_point', 'lab_state_point'] and pd.notna(ann_r_tl['dt']): ax_tl.plot(ann_r_tl['dt'], -0.05, 'rX', markersize=8, label=label_to_use_in_plot); ax_tl.text(ann_r_tl['dt'], -0.07, str(ann_r_tl['annotation'])[:15]+"...", fontsize=7, ha='center', color='darkred'); ev_tl.append(ann_r_tl['dt'])
                elif ann_type == 'interval' and pd.notna(ann_r_tl['dt_start']) and pd.notna(ann_r_tl['dt_end']): ax_tl.plot([ann_r_tl['dt_start'], ann_r_tl['dt_end']], [-0.1, -0.1], lw=6, color='purple', alpha=0.7, label=label_to_use_in_plot); ax_tl.text((ann_r_tl['dt_start'] + ann_r_tl['dt_end'])/2, -0.12, str(ann_r_tl['annotation'])[:15]+"...", fontsize=7, ha='center', color='purple'); ev_tl.extend([ann_r_tl['dt_start'], ann_r_tl['dt_end']])
                elif ann_type == 'clinical_summary': general_annotations_text.append(f"- {ann_r_tl['annotation']} (Rule: {ann_r_tl['rule_id']})")
        min_e_tl=min((e for e in ev_tl if pd.notna(e)),default=-1e3); max_e_tl=max((e for e in ev_tl if pd.notna(e)),default=1e4); ax_tl.set_xlim(min_e_tl-500,max_e_tl+500)
        if general_annotations_text: ax_tl.text(0.01, 0.98, "Clinical Summary Notes:\n" + "\n".join(general_annotations_text), transform=ax_tl.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax_tl.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize='small')
    else: ax_tl.text(0.5,0.5,'No clinical time data',ha='center',va='center',transform=ax_tl.transAxes)
    case_lab_hb = lab_data_abstracted[(lab_data_abstracted['caseid']==case_id)&(lab_data_abstracted['name']=='hb')]
    if not case_lab_hb.empty:
        ax_hb.plot(case_lab_hb['dt'],case_lab_hb['result'],marker='o',ls='-',label='HB Trend',zorder=2)
        if 'hb' in knowledge_base['lab_thresholds']:
            th=knowledge_base['lab_thresholds']['hb']
            if 'normal_low' in th and 'normal_high' in th: ax_hb.axhspan(th['normal_low'],th['normal_high'],color='lightgreen',alpha=0.3,label=f'Normal ({th["normal_low"]}-{th["normal_high"]})',zorder=1)
            if 'critical_low' in th: ax_hb.axhline(th['critical_low'],color='red',ls='--',alpha=0.5,label=f'Critical Low ({th["critical_low"]})',zorder=1)
        sc='hb_state';colors_map_hb=None
        if sc in case_lab_hb.columns:
            uniq_s_hb=case_lab_hb[sc].dropna().unique()
            if len(uniq_s_hb)>0: colors_map_hb=plt.cm.get_cmap('viridis',len(uniq_s_hb))
            for i,s_v_hb in enumerate(uniq_s_hb):
                if pd.notna(s_v_hb)and s_v_hb!='N/A': subs_hb=case_lab_hb[case_lab_hb[sc]==s_v_hb];ax_hb.scatter(subs_hb['dt'],subs_hb['result'],label=f'State: {s_v_hb}',s=80,alpha=0.8,color=colors_map_hb(i) if colors_map_hb else 'blue',edgecolor='black',zorder=3)
        if f_abstract_s_shapes and 'hb_shapes' in f_abstract_s_shapes:
            shps_hb=[s for s in f_abstract_s_shapes.get('hb_shapes',[]) if s['caseid']==case_id]
            for shp_inf_hb in shps_hb:
                sdt_hb=shp_inf_hb['dt_start'];srs_hb=lab_data_abstracted[(lab_data_abstracted['caseid']==case_id)&(lab_data_abstracted['name']=='hb')&(lab_data_abstracted['dt']>=sdt_hb)].sort_values('dt').head(SHAPE_WINDOW_SIZE)
                if not srs_hb.empty and len(srs_hb)==SHAPE_WINDOW_SIZE:ax_hb.plot(srs_hb['dt'],srs_hb['result'],lw=3,ls='--',alpha=0.7,label=f"Shape: {shp_inf_hb['shape_name']}",zorder=4)
        ax_hb.set_title('Trend for HB');ax_hb.set_xlabel("Time (s)");ax_hb.set_ylabel("HB (g/dL)");ax_hb.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize='small');ax_hb.grid(True)
    else: ax_hb.text(0.5,0.5,'No HB data',ha='center',va='center',transform=ax_hb.transAxes)
    case_lab_gluc = lab_data_abstracted[(lab_data_abstracted['caseid']==case_id)&(lab_data_abstracted['name']=='gluc')]
    if not case_lab_gluc.empty:
        ax_gluc.plot(case_lab_gluc['dt'],case_lab_gluc['result'],marker='o',ls='-',label='GLUC Trend',zorder=2)
        if 'gluc' in knowledge_base['lab_thresholds']:
            th_g=knowledge_base['lab_thresholds']['gluc']
            if 'normal_low' in th_g and 'normal_high' in th_g: ax_gluc.axhspan(th_g['normal_low'],th_g['normal_high'],color='lightgreen',alpha=0.3,label=f'Normal ({th_g["normal_low"]}-{th_g["normal_high"]})',zorder=1)
            if 'critical_low' in th_g: ax_gluc.axhline(th_g['critical_low'],color='red',ls='--',alpha=0.5,label=f'Critical Low ({th_g["critical_low"]})',zorder=1)
        sc_g='gluc_state';colors_map_g=None
        if sc_g in case_lab_gluc.columns:
            uniq_s_g=case_lab_gluc[sc_g].dropna().unique()
            if len(uniq_s_g)>0: colors_map_g=plt.cm.get_cmap('viridis',len(uniq_s_g))
            for i,s_v_g in enumerate(uniq_s_g):
                if pd.notna(s_v_g)and s_v_g!='N/A': subs_g=case_lab_gluc[case_lab_gluc[sc_g]==s_v_g];ax_gluc.scatter(subs_g['dt'],subs_g['result'],label=f'State: {s_v_g}',s=80,alpha=0.8,color=colors_map_g(i) if colors_map_g else 'blue',edgecolor='black',zorder=3)
        if f_abstract_s_shapes and 'gluc_shapes' in f_abstract_s_shapes:
            shps_g=[s for s in f_abstract_s_shapes.get('gluc_shapes',[]) if s['caseid']==case_id]
            for shp_inf_g in shps_g:
                sdt_g=shp_inf_g['dt_start'];srs_g=lab_data_abstracted[(lab_data_abstracted['caseid']==case_id)&(lab_data_abstracted['name']=='gluc')&(lab_data_abstracted['dt']>=sdt_g)].sort_values('dt').head(SHAPE_WINDOW_SIZE)
                if not srs_g.empty and len(srs_g)==SHAPE_WINDOW_SIZE:ax_gluc.plot(srs_g['dt'],srs_g['result'],lw=3,ls='--',alpha=0.7,label=f"Shape: {shp_inf_g['shape_name']}",zorder=4)
        ax_gluc.set_title('Trend for GLUC');ax_gluc.set_xlabel("Time (s)");ax_gluc.set_ylabel("GLUC (mg/dL)");ax_gluc.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize='small');ax_gluc.grid(True)
    else: ax_gluc.text(0.5,0.5,'No GLUC data',ha='center',va='center',transform=ax_gluc.transAxes)
    plt.tight_layout(rect=[0,0,0.88,0.96]);
    if save_to_dir: plt.savefig(os.path.join(save_to_dir,f"summary_case_{case_id}.pdf"),bbox_inches='tight', format='pdf'); plt.close(fig)
    else: plt.show()

def plot_frequent_sequences(frequent_patterns_dict, title="Frequent Event Sequences", save_to_dir=None):
    if not frequent_patterns_dict or not isinstance(frequent_patterns_dict, dict): print(f"No valid frequent patterns dict to plot for '{title}'."); return
    patterns = list(frequent_patterns_dict.keys()); counts = list(frequent_patterns_dict.values())
    if not patterns: print(f"No frequent patterns to plot for '{title}'."); return
    sorted_indices = np.argsort(counts)[::-1]; patterns_to_show = [patterns[i] for i in sorted_indices[:15]]; counts_to_show = [counts[i] for i in sorted_indices[:15]]
    pattern_labels = [' -> '.join(map(str,p)) for p in patterns_to_show]
    fig,ax=plt.subplots(figsize=(12,max(5,len(pattern_labels)*0.4)));ax.barh(pattern_labels,counts_to_show,color='skyblue');ax.set_xlabel("Support Count");ax.set_title(title);ax.invert_yaxis();plt.tight_layout()
    if save_to_dir:filename=title.lower().replace(" ","_").replace("(","").replace(")","").replace(":","")+".pdf";plt.savefig(os.path.join(save_to_dir,filename), format='pdf');plt.close(fig)
    else:plt.show()


SAVE_FIGURES_MULTIPANEL = True

if D_prime_clinical is not None and D_prime_lab is not None:
    valid_cases_to_plot = []
    if clinical_df is not None and not clinical_df.empty:
        all_available_case_ids = clinical_df['caseid'].unique()
        for case_id_req in SPECIFIC_CASE_IDS_TO_PLOT: # Use the specific list
            if case_id_req in all_available_case_ids:
                valid_cases_to_plot.append(case_id_req)
            else:
                print(f"Warning: Requested Case ID {case_id_req} not found in clinical_data. Skipping.")
    else:
        print("Clinical data is not available. Cannot validate requested case IDs.")
        # valid_cases_to_plot will remain empty

    if valid_cases_to_plot: # Check if there are any valid cases to plot
        print(f"Preparing to generate multi-panel visualizations for specified cases: {valid_cases_to_plot}.")
        for i, case_id_to_plot in enumerate(valid_cases_to_plot):
            print(f"\nProcessing multi-panel summary for Case ID: {case_id_to_plot} ({i+1}/{len(valid_cases_to_plot)})")
            plot_case_summary_figure(case_id_to_plot, D_prime_clinical, D_prime_lab, Aknowledge_df, mock_K, F_abstract_S,
                                     save_to_dir=output_viz_dir_multipanel if SAVE_FIGURES_MULTIPANEL else None)
    elif not SPECIFIC_CASE_IDS_TO_PLOT: # If the specific list was empty
        print("No specific case IDs were provided for visualization.")
    else: # If specific list had IDs but none were valid
        print("None of the specified case IDs were found in the data, or D_prime data is not available.")
else:
    print("Skipping multi-panel visualizations as D_prime_clinical or D_prime_lab is not available or empty.")

print("\n--- Plotting Frequent Event Pairs (Global) ---")
if F_abstract_P.get('frequent_event_pairs'):
    plot_frequent_sequences(F_abstract_P.get('frequent_event_pairs'), "Frequent Event Pairs (L2)",
                            save_to_dir=output_viz_dir_multipanel if SAVE_FIGURES_MULTIPANEL else None)
print("\n--- Plotting Frequent Event Triplets (Global) ---")
if F_abstract_P.get('frequent_event_triplets'):
    plot_frequent_sequences(F_abstract_P.get('frequent_event_triplets'), "Frequent Event Triplets (L3)",
                             save_to_dir=output_viz_dir_multipanel if SAVE_FIGURES_MULTIPANEL else None)

print("\n--- Full Script Finished ---")