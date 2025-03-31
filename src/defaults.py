
# Default settings

import os
import numpy as np

base_dir = os.path.normpath(os.path.dirname(__file__) + '/..')

# Raw data paths

train_features_path =  base_dir + '/data/raw/train_features.csv'
train_labels_path =  base_dir + '/data/raw/train_labels.csv'

test_features_path =  base_dir + '/data/raw/test_features.csv'
# test_labels_path =  base_dir + '/data/raw/sdoh_test_labels.csv' # New file with test actuals supplied for report arena, not currently used
submission_format_path =  base_dir + '/data/raw/submission_format.csv'

# Where to save predictions, for submission and later analysis 

cv_predictions_path =   base_dir + '/data/processed/cv_predictions.csv'
test_predictions_path =   base_dir + '/data/processed/test_predictions.csv'

# Where to save model

model_save_path = base_dir + '/models/model.pkl'

# Random seed

SEED = 42
np.random.seed(SEED)

# CV settings (required by MAPIE and also used if train is called with cv or cv_predict)

cv_k = 10
cv_scoring = 'neg_root_mean_squared_error'

# Full feature names, for plotting nice reports

feature_names = {
'age_03': 'Age (03)',
'age_12': 'Age (12)',
'urban_03': 'Locality size (03)',
'urban_12': 'Locality size (12)',
'married_03': 'Marital status (03)',
'married_12': 'Marital status (12)',
'n_mar_03': 'No marriages (03)',
'n_mar_12': 'No marriages (12)',
'edu_gru_03': 'Education (03)',
'edu_gru_12': 'Education (12)',
'n_living_child_03': 'No living children (03)',
'n_living_child_12': 'No living children (12)',
'migration_03': 'Lived or worked in the U.S. (03)',
'migration_12': 'Lived or worked in the U.S. (12)',
'glob_hlth_03': 'Self-reported global health (03)',
'glob_hlth_12': 'Self-reported global health (12)',
'adl_dress_03': 'Difficulty getting dressed (03)',
'adl_dress_12': 'Difficulty getting dressed (12)',
'adl_walk_03': 'Difficulty walking (03)',
'adl_walk_12': 'Difficulty walking (12)',
'adl_bath_03': 'Difficulty bathing (03)',
'adl_bath_12': 'Difficulty bathing (12)',
'adl_eat_03': 'Difficulty eating (03)',
'adl_eat_12': 'Difficulty eating (12)',
'adl_bed_03': 'Difficulty getting in and out of bed (03)',
'adl_bed_12': 'Difficulty getting in and out of bed (12)',
'adl_toilet_03': 'Difficulty using the toilet (03)',
'adl_toilet_12': 'Difficulty using the toilet (12)',
'n_adl_03': 'Number of activities of daily living (ADL) limitations (03)',
'n_adl_12': 'Number of activities of daily living (ADL) limitations (12)',
'iadl_money_03': 'Difficulty managing money (03)',
'iadl_money_12': 'Difficulty managing money (12)',
'iadl_meds_03': 'Difficulty taking medications (03)',
'iadl_meds_12': 'Difficulty taking medications (12)',
'iadl_shop_03': 'Difficulty shopping (03)',
'iadl_shop_12': 'Difficulty shopping (12)',
'iadl_meals_03': 'Difficulty preparing a hot meal (03)',
'iadl_meals_12': 'Difficulty preparing a hot meal (12)',
'n_iadl_03': 'No instrumental activities of daily living (IADL) limitations (03)',
'n_iadl_12': 'No instrumental activities of daily living (IADL) limitations (12)',
'depressed_03': 'Felt depressed (03)',
'depressed_12': 'Felt depressed (12)',
'hard_03': 'Felt that everything was an effort (03)',
'hard_12': 'Felt that everything was an effort (12)',
'restless_03': 'Felt that their sleep was restless (03)',
'restless_12': 'Felt that their sleep was restless (12)',
'happy_03': 'Felt happy (03)',
'happy_12': 'Felt happy (12)',
'lonely_03': 'Felt lonely (03)',
'lonely_12': 'Felt lonely (12)',
'enjoy_03': 'Felt that they enjoyed life (03)',
'enjoy_12': 'Felt that they enjoyed life (12)',
'sad_03': 'Felt sad (03)',
'sad_12': 'Felt sad (12)',
'tired_03': 'Felt tired (03)',
'tired_12': 'Felt tired (12)',
'energetic_03': 'Felt they had a lot of energy (03)',
'energetic_12': 'Felt they had a lot of energy (12)',
'n_depr_03': 'Number of CES-D depressive symptoms (03)',
'n_depr_12': 'Number of CES-D depressive symptoms (12)',
'cesd_depressed_03': 'Has 5+ CES-D depressive symptoms (03)',
'cesd_depressed_12': 'Has 5+ CES-D depressive symptoms (12)',
'hypertension_03': 'Diagnosed with hypertension (03)',
'hypertension_12': 'Diagnosed with hypertension (12)',
'diabetes_03': 'Diagnosed with diabetes (03)',
'diabetes_12': 'Diagnosed with diabetes (12)',
'resp_ill_03': 'Diagnosed with respiratory illness (03)',
'resp_ill_12': 'Diagnosed with respiratory illness (12)',
'arthritis_03': 'Diagnosed with arthritis/rheumatism (03)',
'arthritis_12': 'Diagnosed with arthritis/rheumatism (12)',
'hrt_attack_03': 'Has been told they had a heart attack (03)',
'hrt_attack_12': 'Has been told they had a heart attack (12)',
'stroke_03': 'Has been told they had a stroke (03)',
'stroke_12': 'Has been told they had a stroke (12)',
'cancer_03': 'Diagnosed with cancer (03)',
'cancer_12': 'Diagnosed with cancer (12)',
'n_illnesses_03': 'No illnesses (03)',
'n_illnesses_12': 'No Illnesses (12)',
'bmi_03': 'BMI (03)',
'bmi_12': 'BMI (12)',
'exer_3xwk_03': 'Exercises 3+ times per week (03)',
'exer_3xwk_12': 'Exercises 3+ times per week (12)',
'alcohol_03': 'Alcohol (03)',
'alcohol_12': 'Alcohol (12)',
'tobacco_03': 'Tobacco (03)',
'tobacco_12': 'Tobacco (12)',
'test_chol_03': 'Cholesterol blood test (03)',
'test_chol_12': 'Cholesterol blood test (12)',
'test_tuber_03': 'Tested for tuberculosis (03)',
'test_tuber_12': 'Tested for tuberculosis (12)',
'test_diab_03': 'Tested for diabetes (03)',
'test_diab_12': 'Tested for diabetes (12)',
'test_pres_03': 'Tested for high blood pressure (03)',
'test_pres_12': 'Tested for high blood pressure (12)',
'hosp_03': 'Has been hospitalized (03)',
'hosp_12': 'Has been hospitalized (12)',
'visit_med_03': 'Visited doctor in last year (03)',
'visit_med_12': 'Visited doctor in last year (12)',
'out_proc_03': 'Outpatient procedure in last year (03)',
'out_proc_12': 'Outpatient procedure in last year (12)',
'visit_dental_03': 'Visited dentist in last year (03)',
'visit_dental_12': 'Visited dentist in last year (12)',
'imss_03': 'Health coverage with IMSS (03)',
'imss_12': 'Health coverage with IMSS (12)',
'issste_03': 'Health coverage with ISSSTE/ISSSTE Estatal (03)',
'issste_12': 'Health coverage with ISSSTE/ISSSTE Estatal (12)',
'pem_def_mar_03': 'Health coverage with PEMEX, Defensa, or Marina (03)',
'pem_def_mar_12': 'Health coverage with PEMEX, Defensa, or Marina (12)',
'insur_private_03': 'Health coverage with private health insurance (03)',
'insur_private_12': 'Health coverage with private health insurance (12)',
'insur_other_03': 'Health coverage with other health insurance (03)',
'insur_other_12': 'Health coverage with other health insurance (12)',
'seg_pop_12': 'Health coverage with Seguro Popular (12)',
'insured_03': 'Has health insurance (03)',
'insured_12': 'Has health insurance (12)',
'decis_famil_03': 'Weight in family decisions (03)',
'decis_famil_12': 'Weight in family decisions (12)',
'decis_personal_03': 'Weight over personal decisions (03)',
'decis_personal_12': 'Weight over personal decisions (12)',
'employment_03': 'Employment status (03)',
'employment_12': 'Employment status (12)',
'vax_flu_12': 'Has been vaccinated against flu (12)',
'vax_pneu_12': 'Has been vaccinated against pneumonia (12)',
'care_adult_12': 'Uses time to look after a sick or disabled adult (12)',
'care_child_12': 'Uses time to look after children under 12 (12)',
'volunteer_12': 'Uses time to volunteer for a non-profit (12)',
'attends_class_12': 'Uses time to attend training course, lecture, or class (12)',
'attends_club_12': 'Uses time to attend sports or social club (12)',
'reads_12': 'Read books, etc. (12)',
'games_12': 'Crosswords, puzzles, etc. (12)',
'table_games_12': 'Tabletop games (12)',
'comms_tel_comp_12': 'Talk on the phone, send message, use the web (12)',
'act_mant_12': 'Maintain a house, do repairs, garden (12)',
'tv_12': 'Watch television (12)',
'sewing_12': 'Sew, emboider, etc. (12)',
'satis_ideal_12': 'Life close to ideal (12)',
'satis_excel_12': 'Life excellent (12)',
'satis_fine_12': 'Satisfied with life (12)',
'cosas_imp_12': 'Achieved the things in life that are important (12)',
'wouldnt_change_12': 'Would change almost nothing about their life (12)',
'memory_12': 'Self-reported memory (12)',
'ragender': 'Gender',
'rameduc_m': 'Mother\'s education',
'rafeduc_m': 'Father\'s education',
'sgender_03': 'Spouse gender (03)',
'sgender_12': 'Spouse gender (12)',
'rjob_hrswk_03': 'Hours per week at main job (03)',
'rjob_hrswk_12': 'Hours per week at main job (12)',
'rjlocc_m_03': 'Category of longest occuptation (03)',
'rjlocc_m_12': 'Category of longest occuptation (12)',
'rjob_end_03': 'Year last job ended (03)',
'rjob_end_12': 'Year last job ended (12)',
'rjobend_reason_03': 'Reason last job ended (03)',
'rjobend_reason_12': 'Reason last job ended (12)',
'rearnings_03': 'Earnings from employment (03)',
'rearnings_12': 'Earnings from employment (12)',
'searnings_03': 'Spouse''s earnings from employment (03)',
'searnings_12': 'Spouse''s earnings from employment (12)',
'hincome_03': 'Household income (03)',
'hincome_12': 'Household income (12)',
'hinc_business_03': 'Household income from business (03)',
'hinc_business_12': 'Household income from business (12)',
'hinc_rent_03': 'Household income from rent (03)',
'hinc_rent_12': 'Household income from rent (12)',
'hinc_assets_03': 'Household income from financial assets (03)',
'hinc_assets_12': 'Household income from financial assets (12)',
'hinc_cap_03': 'Household capital income (03)',
'hinc_cap_12': 'Household capital income (12)',
'rinc_pension_03': 'Income from pensions (03)',
'rinc_pension_12': 'Income from pensions (12)',
'sinc_pension_03': 'Spouse\'s income from pensions (03)',
'sinc_pension_12': 'Spouse\'s income from pensions (12)',
'rrelgimp_03': 'Importance of religion (03)',
'rrelgimp_12': 'Importance of religion (12)',
'rrfcntx_m_12': 'How often see friends and relatives (12)',
'rsocact_m_12': 'How often have social activities (12)',
'rrelgwk_12': 'Participates in weekly religious services (12)',
'a16a_12': 'Year when respondent first left for the U.S., if they ever lived in the U.S.',
'a21_12': 'Total years lived or worked in the U.S.',
'a22_12': 'Main job type during longest stay in the U.S.',
'a33b_12': 'U.S. residency status',
'a34_12': 'Speaks English',
'j11_12': 'Floor material of residence',
'16_assessment': 'Had 2016 assessment',
'21_assessment': 'Had 2021 assessment',
'03_survey': 'Did 03 survey',
'12_survey': 'Did 12 survey',
'year': 'Prediction year'
}

cmd_help_message = 'Please try one of:\n  python predict.py --help\n  python train.py --help'

#------------------------------------------------------------------------------

if __name__ == '__main__':
    print(cmd_help_message)