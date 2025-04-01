import numpy as np
import pandas as pd

import src.defaults as defaults

#------------------------------------------------------------------------------

# 2003 survey features
features_03 = [   
    'age_03', 'urban_03', 'married_03', 'n_mar_03', 'edu_gru_03', 'n_living_child_03', 
    'migration_03', 'glob_hlth_03', 'adl_dress_03', 'adl_walk_03', 'adl_bath_03', 
    'adl_eat_03', 'adl_bed_03', 'adl_toilet_03', 'n_adl_03', 'iadl_money_03', 'iadl_meds_03', 
    'iadl_shop_03', 'iadl_meals_03', 'n_iadl_03', 'depressed_03', 'hard_03', 'restless_03', 
    'happy_03', 'lonely_03', 'enjoy_03', 'sad_03', 'tired_03', 'energetic_03', 'n_depr_03', 
    'cesd_depressed_03', 'hypertension_03', 'diabetes_03', 'resp_ill_03', 'arthritis_03', 
    'hrt_attack_03', 'stroke_03', 'cancer_03', 'n_illnesses_03', 'bmi_03', 'exer_3xwk_03', 
    'alcohol_03', 'tobacco_03', 'test_chol_03', 'test_tuber_03', 'test_diab_03', 'test_pres_03', 
    'hosp_03', 'visit_med_03', 'out_proc_03', 'visit_dental_03', 'imss_03', 'issste_03', 
    'pem_def_mar_03', 'insur_private_03', 'insur_other_03', 'insured_03', 'decis_famil_03', 
    'decis_personal_03', 'employment_03', 'sgender_03', 'rjob_hrswk_03', 'rjlocc_m_03', 
    'rjob_end_03', 'rjobend_reason_03', 'rearnings_03', 'searnings_03', 'hincome_03', 
    'hinc_business_03', 'hinc_rent_03', 'hinc_assets_03', 'hinc_cap_03', 'rinc_pension_03', 
    'sinc_pension_03', 'rrelgimp_03'
]

# 2012 survey features
features_12 = [  
    'age_12', 'urban_12', 'married_12', 'n_mar_12', 'edu_gru_12', 'n_living_child_12', 
    'migration_12', 'glob_hlth_12', 'adl_dress_12', 'adl_walk_12', 'adl_bath_12', 'adl_eat_12', 
    'adl_bed_12', 'adl_toilet_12', 'n_adl_12', 'iadl_money_12', 'iadl_meds_12', 'iadl_shop_12', 
    'iadl_meals_12', 'n_iadl_12', 'depressed_12', 'hard_12', 'restless_12', 'happy_12', 'lonely_12', 
    'enjoy_12', 'sad_12', 'tired_12', 'energetic_12', 'n_depr_12', 'cesd_depressed_12', 
    'hypertension_12', 'diabetes_12', 'resp_ill_12', 'arthritis_12', 'hrt_attack_12', 'stroke_12', 
    'cancer_12', 'n_illnesses_12', 'bmi_12', 'exer_3xwk_12', 'alcohol_12', 'tobacco_12', 
    'test_chol_12', 'test_tuber_12', 'test_diab_12', 'test_pres_12', 'hosp_12', 'visit_med_12', 
    'out_proc_12', 'visit_dental_12', 'imss_12', 'issste_12', 'pem_def_mar_12', 'insur_private_12', 
    'insur_other_12', 'insured_12', 'decis_famil_12', 'decis_personal_12', 'employment_12', 
    'vax_flu_12', 'vax_pneu_12', 'seg_pop_12', 'care_adult_12', 'care_child_12', 'volunteer_12', 
    'attends_class_12', 'attends_club_12', 'reads_12', 'games_12', 'table_games_12', 
    'comms_tel_comp_12', 'act_mant_12', 'tv_12', 'sewing_12', 'satis_ideal_12', 'satis_excel_12', 
    'satis_fine_12', 'cosas_imp_12', 'wouldnt_change_12', 'memory_12', 'sgender_12', 'rjob_hrswk_12', 
    'rjlocc_m_12', 'rjob_end_12', 'rjobend_reason_12', 'rearnings_12', 'searnings_12', 'hincome_12', 
    'hinc_business_12', 'hinc_rent_12', 'hinc_assets_12', 'hinc_cap_12', 'rinc_pension_12', 
    'sinc_pension_12', 'rrelgimp_12', 'rrfcntx_m_12', 'rsocact_m_12', 'rrelgwk_12', 'a16a_12', 
    'a21_12', 'a22_12', 'a33b_12', 'a34_12', 'j11_12'
]

# Features with more than 66% missing data in train set
# Removing these showed a (very small) improvement in performance
# Debateable if helpful, but these were applied to our best performing model
# so have been retained to ensure the predictions are replicable.
features_to_drop = [
    'rjlocc_m_03', 'rjob_end_03', 'rjobend_reason_03', 'rjob_end_12', 'rjobend_reason_12', 
    'a16a_12', 'a21_12', 'a22_12', 'a33b_12'
]

#------------------------------------------------------------------------------

def prepare_data(features, labels):
    """Prepares data for training or prediction.
    
    Transforms features and labels into X, y and uids for training and prediction.
    In the case of test, pass the submission format data as the labels,
    as this contains the uid-years we need to predict for.
    The returned uids should be passed as groups for cross validation.

    Args:
        features (pd.DataFrame): train, test or prediction features
        labels (pd.DataFrame): train labels, or in the case of test, the submission format
        
    Returns:
        X, y, uids
    """
    
    # All ordinal cols apart from a34_12 begin with an ordinal number, which is helpful
    # So fix values in that column
    features['a34_12'] = features['a34_12'].replace({'Yes 1':'1. Yes', 'No 2': '0. No'})
    
    # decis_personal_* loads as an int for 03 and object for 12
    # Make this consistent
    r = {'1. A lot': '1', '2. A little': '2', '3. None': '3'}
    features['decis_personal_12'] = features['decis_personal_12'].replace(r).astype('float')
    
    # Convert object to category columns
    for col in features.select_dtypes(include='object'):
        if col == 'uid': continue
        features[col] = features[col].astype('category')

    # Add features to record whether individual completed 16 and 21 assessments
    uid2016 = labels[labels['year'] == 2016]['uid']
    uid2021 = labels[labels['year'] == 2021]['uid']
    features['16_assessment'] = features['uid'].isin(uid2016)
    features['21_assessment'] = features['uid'].isin(uid2021)
    
    # Add features to record whether individual did 03 and 12 surveys
    features['03_survey'] = features[features_03].notna().any(axis=1)
    features['12_survey'] = features[features_12].notna().any(axis=1)

    # Remove features we don't want to use
    features.drop(columns = features_to_drop, inplace=True)
    
    # Catboost does not support NaNs in categories, so replace with 'Missing'
    for col in features.select_dtypes(include='category'):
        features[col] = features[col].astype('object').fillna('Missing').astype('category')
        
    # Merge features and labels
    merged = features.merge(labels, on='uid', how='left')
    
    # Create X, y, uids from merged    
    X = merged.drop(columns = ['uid','composite_score'])
    y = merged['composite_score']
    uids = merged['uid']

    return X, y, uids

#------------------------------------------------------------------------------

if __name__ == '__main__':
    print(defaults.cmd_help_message)