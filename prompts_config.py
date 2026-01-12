PROMPTS = {
    "ckd": {
        "acquisition_prompt": (
            "You are an expert diagnostician for Chronic Kidney Disease (CKD).\n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Feature explanation: Age: The age of the patients ranges from 20 to 90 years. Gender: Gender of the patients, where 0 represents Male and 1 represents Female. SocioeconomicStatus: The socioeconomic status of the patients. EducationLevel: The education level of the patients, coded as follows:0: None,1: High School,2: Bachelor's,3: Higher. BMI: Body Mass Index of the patients, ranging from 15 to 40. Smoking: Smoking status, where 0 indicates No and 1 indicates Yes. AlcoholConsumption: Weekly alcohol consumption in units, ranging from 0 to 20. PhysicalActivity: Weekly physical activity in hours, ranging from 0 to 10. DietQuality: Diet quality score, ranging from 0 to 10. SleepQuality: Sleep quality score, ranging from 4 to 10. FamilyHistoryKidneyDisease: Family history of kidney disease, where 0 indicates No and 1 indicates Yes. FamilyHistoryHypertension: Family history of hypertension, where 0 indicates No and 1 indicates Yes. FamilyHistoryDiabetes: Family history of diabetes, where 0 indicates No and 1 indicates Yes. PreviousAcuteKidneyInjury: History of previous acute kidney injury, where 0 indicates No and 1 indicates Yes. UrinaryTractInfections: History of urinary tract infections, where 0 indicates No and 1 indicates Yes. SystolicBP: Systolic blood pressure, ranging from 90 to 180 mmHg. DiastolicBP: Diastolic blood pressure, ranging from 60 to 120 mmHg. FastingBloodSugar: Fasting blood sugar levels, ranging from 70 to 200 mg/dL. HbA1c: Hemoglobin A1c levels, ranging from 4.0% to 10.0%. SerumCreatinine: Serum creatinine levels, ranging from 0.5 to 5.0 mg/dL. BUNLevels: Blood Urea Nitrogen levels, ranging from 5 to 50 mg/dL. GFR: Glomerular Filtration Rate, ranging from 15 to 120 mL/min/1.73 m². ProteinInUrine: Protein levels in urine, ranging from 0 to 5 g/day. ACR: Albumin-to-Creatinine Ratio, ranging from 0 to 300 mg/g. SerumElectrolytesSodium: Serum sodium levels, ranging from 135 to 145 mEq/L. SerumElectrolytesPotassium: Serum potassium levels, ranging from 3.5 to 5.5 mEq/L. SerumElectrolytesCalcium: Serum calcium levels, ranging from 8.5 to 10.5 mg/dL. SerumElectrolytesPhosphorus: Serum phosphorus levels, ranging from 2.5 to 4.5 mg/dL. HemoglobinLevels: Hemoglobin levels, ranging from 10 to 18 g/dL. CholesterolTotal: Total cholesterol levels, ranging from 150 to 300 mg/dL. CholesterolLDL: Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL. CholesterolHDL: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL. CholesterolTriglycerides: Triglycerides levels, ranging from 50 to 400 mg/dL. ACEInhibitors: Use of ACE inhibitors, where 0 indicates No and 1 indicates Yes. Diuretics: Use of diuretics, where 0 indicates No and 1 indicates Yes. NSAIDsUse: Frequency of NSAIDs use, ranging from 0 to 10 times per week. Statins: Use of statins, where 0 indicates No and 1 indicates Yes. AntidiabeticMedications: Use of antidiabetic medications, where 0 indicates No and 1 indicates Yes. Edema: Presence of edema, where 0 indicates No and 1 indicates Yes. FatigueLevels: Fatigue levels, ranging from 0 to 10. NauseaVomiting: Frequency of nausea and vomiting, ranging from 0 to 7 times per week. MuscleCramps: Frequency of muscle cramps, ranging from 0 to 7 times per week. Itching: Itching severity, ranging from 0 to 10. QualityOfLifeScore: Quality of life score, ranging from 0 to 100. HeavyMetalsExposure: Exposure to heavy metals, where 0 indicates No and 1 indicates Yes. OccupationalExposureChemicals: Occupational exposure to harmful chemicals, where 0 indicates No and 1 indicates Yes. WaterQuality: Quality of water, where 0 indicates Good and 1 indicates Poor. MedicalCheckupsFrequency: Frequency of medical check-ups per year, ranging from 0 to 4. MedicationAdherence: Medication adherence score, ranging from 0 to 10. HealthLiteracy: Health literacy score, ranging from 0 to 10.\n"
            "Analyze based on the similar historical cases, the current patient features, and your knowledge in the subject matter. Which SINGLE feature should be acquired next to improve diagnosis? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0": (
            "You are an expert diagnostician for Chronic Kidney Disease (CKD).\n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Feature explanation: Age: The age of the patients ranges from 20 to 90 years. Gender: Gender of the patients, where 0 represents Male and 1 represents Female. SocioeconomicStatus: The socioeconomic status of the patients. EducationLevel: The education level of the patients, coded as follows:0: None,1: High School,2: Bachelor's,3: Higher. BMI: Body Mass Index of the patients, ranging from 15 to 40. Smoking: Smoking status, where 0 indicates No and 1 indicates Yes. AlcoholConsumption: Weekly alcohol consumption in units, ranging from 0 to 20. PhysicalActivity: Weekly physical activity in hours, ranging from 0 to 10. DietQuality: Diet quality score, ranging from 0 to 10. SleepQuality: Sleep quality score, ranging from 4 to 10. FamilyHistoryKidneyDisease: Family history of kidney disease, where 0 indicates No and 1 indicates Yes. FamilyHistoryHypertension: Family history of hypertension, where 0 indicates No and 1 indicates Yes. FamilyHistoryDiabetes: Family history of diabetes, where 0 indicates No and 1 indicates Yes. PreviousAcuteKidneyInjury: History of previous acute kidney injury, where 0 indicates No and 1 indicates Yes. UrinaryTractInfections: History of urinary tract infections, where 0 indicates No and 1 indicates Yes. SystolicBP: Systolic blood pressure, ranging from 90 to 180 mmHg. DiastolicBP: Diastolic blood pressure, ranging from 60 to 120 mmHg. FastingBloodSugar: Fasting blood sugar levels, ranging from 70 to 200 mg/dL. HbA1c: Hemoglobin A1c levels, ranging from 4.0% to 10.0%. SerumCreatinine: Serum creatinine levels, ranging from 0.5 to 5.0 mg/dL. BUNLevels: Blood Urea Nitrogen levels, ranging from 5 to 50 mg/dL. GFR: Glomerular Filtration Rate, ranging from 15 to 120 mL/min/1.73 m². ProteinInUrine: Protein levels in urine, ranging from 0 to 5 g/day. ACR: Albumin-to-Creatinine Ratio, ranging from 0 to 300 mg/g. SerumElectrolytesSodium: Serum sodium levels, ranging from 135 to 145 mEq/L. SerumElectrolytesPotassium: Serum potassium levels, ranging from 3.5 to 5.5 mEq/L. SerumElectrolytesCalcium: Serum calcium levels, ranging from 8.5 to 10.5 mg/dL. SerumElectrolytesPhosphorus: Serum phosphorus levels, ranging from 2.5 to 4.5 mg/dL. HemoglobinLevels: Hemoglobin levels, ranging from 10 to 18 g/dL. CholesterolTotal: Total cholesterol levels, ranging from 150 to 300 mg/dL. CholesterolLDL: Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL. CholesterolHDL: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL. CholesterolTriglycerides: Triglycerides levels, ranging from 50 to 400 mg/dL. ACEInhibitors: Use of ACE inhibitors, where 0 indicates No and 1 indicates Yes. Diuretics: Use of diuretics, where 0 indicates No and 1 indicates Yes. NSAIDsUse: Frequency of NSAIDs use, ranging from 0 to 10 times per week. Statins: Use of statins, where 0 indicates No and 1 indicates Yes. AntidiabeticMedications: Use of antidiabetic medications, where 0 indicates No and 1 indicates Yes. Edema: Presence of edema, where 0 indicates No and 1 indicates Yes. FatigueLevels: Fatigue levels, ranging from 0 to 10. NauseaVomiting: Frequency of nausea and vomiting, ranging from 0 to 7 times per week. MuscleCramps: Frequency of muscle cramps, ranging from 0 to 7 times per week. Itching: Itching severity, ranging from 0 to 10. QualityOfLifeScore: Quality of life score, ranging from 0 to 100. HeavyMetalsExposure: Exposure to heavy metals, where 0 indicates No and 1 indicates Yes. OccupationalExposureChemicals: Occupational exposure to harmful chemicals, where 0 indicates No and 1 indicates Yes. WaterQuality: Quality of water, where 0 indicates Good and 1 indicates Poor. MedicalCheckupsFrequency: Frequency of medical check-ups per year, ranging from 0 to 4. MedicationAdherence: Medication adherence score, ranging from 0 to 10. HealthLiteracy: Health literacy score, ranging from 0 to 10.\n"
            "Analyze based on your knowledge in the subject matter and the current patient features. Which SINGLE feature should be acquired next to improve diagnosis? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt": (
            "You are an expert diagnostician for Chronic Kidney Disease (CKD).\n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Feature explanation: Age: The age of the patients ranges from 20 to 90 years. Gender: Gender of the patients, where 0 represents Male and 1 represents Female. SocioeconomicStatus: The socioeconomic status of the patients. EducationLevel: The education level of the patients, coded as follows:0: None,1: High School,2: Bachelor's,3: Higher. BMI: Body Mass Index of the patients, ranging from 15 to 40. Smoking: Smoking status, where 0 indicates No and 1 indicates Yes. AlcoholConsumption: Weekly alcohol consumption in units, ranging from 0 to 20. PhysicalActivity: Weekly physical activity in hours, ranging from 0 to 10. DietQuality: Diet quality score, ranging from 0 to 10. SleepQuality: Sleep quality score, ranging from 4 to 10. FamilyHistoryKidneyDisease: Family history of kidney disease, where 0 indicates No and 1 indicates Yes. FamilyHistoryHypertension: Family history of hypertension, where 0 indicates No and 1 indicates Yes. FamilyHistoryDiabetes: Family history of diabetes, where 0 indicates No and 1 indicates Yes. PreviousAcuteKidneyInjury: History of previous acute kidney injury, where 0 indicates No and 1 indicates Yes. UrinaryTractInfections: History of urinary tract infections, where 0 indicates No and 1 indicates Yes. SystolicBP: Systolic blood pressure, ranging from 90 to 180 mmHg. DiastolicBP: Diastolic blood pressure, ranging from 60 to 120 mmHg. FastingBloodSugar: Fasting blood sugar levels, ranging from 70 to 200 mg/dL. HbA1c: Hemoglobin A1c levels, ranging from 4.0% to 10.0%. SerumCreatinine: Serum creatinine levels, ranging from 0.5 to 5.0 mg/dL. BUNLevels: Blood Urea Nitrogen levels, ranging from 5 to 50 mg/dL. GFR: Glomerular Filtration Rate, ranging from 15 to 120 mL/min/1.73 m². ProteinInUrine: Protein levels in urine, ranging from 0 to 5 g/day. ACR: Albumin-to-Creatinine Ratio, ranging from 0 to 300 mg/g. SerumElectrolytesSodium: Serum sodium levels, ranging from 135 to 145 mEq/L. SerumElectrolytesPotassium: Serum potassium levels, ranging from 3.5 to 5.5 mEq/L. SerumElectrolytesCalcium: Serum calcium levels, ranging from 8.5 to 10.5 mg/dL. SerumElectrolytesPhosphorus: Serum phosphorus levels, ranging from 2.5 to 4.5 mg/dL. HemoglobinLevels: Hemoglobin levels, ranging from 10 to 18 g/dL. CholesterolTotal: Total cholesterol levels, ranging from 150 to 300 mg/dL. CholesterolLDL: Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL. CholesterolHDL: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL. CholesterolTriglycerides: Triglycerides levels, ranging from 50 to 400 mg/dL. ACEInhibitors: Use of ACE inhibitors, where 0 indicates No and 1 indicates Yes. Diuretics: Use of diuretics, where 0 indicates No and 1 indicates Yes. NSAIDsUse: Frequency of NSAIDs use, ranging from 0 to 10 times per week. Statins: Use of statins, where 0 indicates No and 1 indicates Yes. AntidiabeticMedications: Use of antidiabetic medications, where 0 indicates No and 1 indicates Yes. Edema: Presence of edema, where 0 indicates No and 1 indicates Yes. FatigueLevels: Fatigue levels, ranging from 0 to 10. NauseaVomiting: Frequency of nausea and vomiting, ranging from 0 to 7 times per week. MuscleCramps: Frequency of muscle cramps, ranging from 0 to 7 times per week. Itching: Itching severity, ranging from 0 to 10. QualityOfLifeScore: Quality of life score, ranging from 0 to 100. HeavyMetalsExposure: Exposure to heavy metals, where 0 indicates No and 1 indicates Yes. OccupationalExposureChemicals: Occupational exposure to harmful chemicals, where 0 indicates No and 1 indicates Yes. WaterQuality: Quality of water, where 0 indicates Good and 1 indicates Poor. MedicalCheckupsFrequency: Frequency of medical check-ups per year, ranging from 0 to 4. MedicationAdherence: Medication adherence score, ranging from 0 to 10. HealthLiteracy: Health literacy score, ranging from 0 to 10.\n"
            "Analyze based on your knowledge in the subject matter, the similar historical cases, the current patient features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt_k0": (
            "You are an expert diagnostician for Chronic Kidney Disease (CKD).\n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Feature explanation: Age: The age of the patients ranges from 20 to 90 years. Gender: Gender of the patients, where 0 represents Male and 1 represents Female. SocioeconomicStatus: The socioeconomic status of the patients. EducationLevel: The education level of the patients, coded as follows:0: None,1: High School,2: Bachelor's,3: Higher. BMI: Body Mass Index of the patients, ranging from 15 to 40. Smoking: Smoking status, where 0 indicates No and 1 indicates Yes. AlcoholConsumption: Weekly alcohol consumption in units, ranging from 0 to 20. PhysicalActivity: Weekly physical activity in hours, ranging from 0 to 10. DietQuality: Diet quality score, ranging from 0 to 10. SleepQuality: Sleep quality score, ranging from 4 to 10. FamilyHistoryKidneyDisease: Family history of kidney disease, where 0 indicates No and 1 indicates Yes. FamilyHistoryHypertension: Family history of hypertension, where 0 indicates No and 1 indicates Yes. FamilyHistoryDiabetes: Family history of diabetes, where 0 indicates No and 1 indicates Yes. PreviousAcuteKidneyInjury: History of previous acute kidney injury, where 0 indicates No and 1 indicates Yes. UrinaryTractInfections: History of urinary tract infections, where 0 indicates No and 1 indicates Yes. SystolicBP: Systolic blood pressure, ranging from 90 to 180 mmHg. DiastolicBP: Diastolic blood pressure, ranging from 60 to 120 mmHg. FastingBloodSugar: Fasting blood sugar levels, ranging from 70 to 200 mg/dL. HbA1c: Hemoglobin A1c levels, ranging from 4.0% to 10.0%. SerumCreatinine: Serum creatinine levels, ranging from 0.5 to 5.0 mg/dL. BUNLevels: Blood Urea Nitrogen levels, ranging from 5 to 50 mg/dL. GFR: Glomerular Filtration Rate, ranging from 15 to 120 mL/min/1.73 m². ProteinInUrine: Protein levels in urine, ranging from 0 to 5 g/day. ACR: Albumin-to-Creatinine Ratio, ranging from 0 to 300 mg/g. SerumElectrolytesSodium: Serum sodium levels, ranging from 135 to 145 mEq/L. SerumElectrolytesPotassium: Serum potassium levels, ranging from 3.5 to 5.5 mEq/L. SerumElectrolytesCalcium: Serum calcium levels, ranging from 8.5 to 10.5 mg/dL. SerumElectrolytesPhosphorus: Serum phosphorus levels, ranging from 2.5 to 4.5 mg/dL. HemoglobinLevels: Hemoglobin levels, ranging from 10 to 18 g/dL. CholesterolTotal: Total cholesterol levels, ranging from 150 to 300 mg/dL. CholesterolLDL: Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL. CholesterolHDL: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL. CholesterolTriglycerides: Triglycerides levels, ranging from 50 to 400 mg/dL. ACEInhibitors: Use of ACE inhibitors, where 0 indicates No and 1 indicates Yes. Diuretics: Use of diuretics, where 0 indicates No and 1 indicates Yes. NSAIDsUse: Frequency of NSAIDs use, ranging from 0 to 10 times per week. Statins: Use of statins, where 0 indicates No and 1 indicates Yes. AntidiabeticMedications: Use of antidiabetic medications, where 0 indicates No and 1 indicates Yes. Edema: Presence of edema, where 0 indicates No and 1 indicates Yes. FatigueLevels: Fatigue levels, ranging from 0 to 10. NauseaVomiting: Frequency of nausea and vomiting, ranging from 0 to 7 times per week. MuscleCramps: Frequency of muscle cramps, ranging from 0 to 7 times per week. Itching: Itching severity, ranging from 0 to 10. QualityOfLifeScore: Quality of life score, ranging from 0 to 100. HeavyMetalsExposure: Exposure to heavy metals, where 0 indicates No and 1 indicates Yes. OccupationalExposureChemicals: Occupational exposure to harmful chemicals, where 0 indicates No and 1 indicates Yes. WaterQuality: Quality of water, where 0 indicates Good and 1 indicates Poor. MedicalCheckupsFrequency: Frequency of medical check-ups per year, ranging from 0 to 4. MedicationAdherence: Medication adherence score, ranging from 0 to 10. HealthLiteracy: Health literacy score, ranging from 0 to 10.\n"
            "Analyze based on your knowledge in the subject matter, the current patient features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        )
    },
    "ctgs": {
        "acquisition_prompt": (
            "You are an expert diagnostician for aids. You are tasked to predict the censoring indicator of the AIDS patient. 1 means failure, 0 means censoring. \n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Feature explanation: time: time to failure or censoring. trt: treatment indicator (0 = ZDV only; 1 = ZDV + ddl, 2 = ZDV + Zal, 3 = ddl only). age: age (yrs) at baseline. wtkg: weight (kg) at baseline. hemo: hemophilia (0=no, 1=yes). homo: sexual orientation, homosexual activity (0=no, 1=yes). drugs: history of IV drug use (0=no, 1=yes). karnof: Karnofsky score (on a scale of 0-100). oprior: Non-ZDV antiretroviral therapy pre-175 (0=no, 1=yes). z30: ZDV in the 30 days prior to 175 (0=no, 1=yes). zprior: ZDV prior to 175 (0=no, 1=yes). preanti: number of days pre-175 anti-retroviral therapy. race: race (0=White, 1=non-white). gender: gender (0=F, 1=M). str2: antiretroviral history (0=naive, 1=experienced). strat: antiretroviral history stratification (1='Antiretroviral Naive',2='> 1 but <= 52 weeks of prior. symptom: symptomatic indicator (0=asymp, 1=symp). treat: treatment indicator (0=ZDV only, 1=others). offtrt: indicator of off-trt before 96+/-5 weeks (0=no,1=yes). cd40: CD4 at baseline. cd420: CD4 at 20+/-5 weeks. cd80: CD8 at baseline. cd820: CD8 at 20+/-5 weeks.\n"
            "Analyze based on the similar historical cases, the current patient features, and your knowledge in the subject matter. Which SINGLE feature should be acquired next to improve diagnosis? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0": (
            "You are an expert diagnostician for aids. You are tasked to predict the censoring indicator of the AIDS patient. 1 means failure, 0 means censoring. \n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Feature explanation: time: time to failure or censoring. trt: treatment indicator (0 = ZDV only; 1 = ZDV + ddl, 2 = ZDV + Zal, 3 = ddl only). age: age (yrs) at baseline. wtkg: weight (kg) at baseline. hemo: hemophilia (0=no, 1=yes). homo: sexual orientation, homosexual activity (0=no, 1=yes). drugs: history of IV drug use (0=no, 1=yes). karnof: Karnofsky score (on a scale of 0-100). oprior: Non-ZDV antiretroviral therapy pre-175 (0=no, 1=yes). z30: ZDV in the 30 days prior to 175 (0=no, 1=yes). zprior: ZDV prior to 175 (0=no, 1=yes). preanti: number of days pre-175 anti-retroviral therapy. race: race (0=White, 1=non-white). gender: gender (0=F, 1=M). str2: antiretroviral history (0=naive, 1=experienced). strat: antiretroviral history stratification (1='Antiretroviral Naive',2='> 1 but <= 52 weeks of prior. symptom: symptomatic indicator (0=asymp, 1=symp). treat: treatment indicator (0=ZDV only, 1=others). offtrt: indicator of off-trt before 96+/-5 weeks (0=no,1=yes). cd40: CD4 at baseline. cd420: CD4 at 20+/-5 weeks. cd80: CD8 at baseline. cd820: CD8 at 20+/-5 weeks.\n"
            "Analyze based on your knowledge in the subject matter and the current patient features. Which SINGLE feature should be acquired next to improve diagnosis? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt": (
            "You are an expert diagnostician for Cardiotocography (CTGS).\n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Feature explanation: time: time to failure or censoring. trt: treatment indicator (0 = ZDV only; 1 = ZDV + ddl, 2 = ZDV + Zal, 3 = ddl only). age: age (yrs) at baseline. wtkg: weight (kg) at baseline. hemo: hemophilia (0=no, 1=yes). homo: sexual orientation, homosexual activity (0=no, 1=yes). drugs: history of IV drug use (0=no, 1=yes). karnof: Karnofsky score (on a scale of 0-100). oprior: Non-ZDV antiretroviral therapy pre-175 (0=no, 1=yes). z30: ZDV in the 30 days prior to 175 (0=no, 1=yes). zprior: ZDV prior to 175 (0=no, 1=yes). preanti: number of days pre-175 anti-retroviral therapy. race: race (0=White, 1=non-white). gender: gender (0=F, 1=M). str2: antiretroviral history (0=naive, 1=experienced). strat: antiretroviral history stratification (1='Antiretroviral Naive',2='> 1 but <= 52 weeks of prior. symptom: symptomatic indicator (0=asymp, 1=symp). treat: treatment indicator (0=ZDV only, 1=others). offtrt: indicator of off-trt before 96+/-5 weeks (0=no,1=yes). cd40: CD4 at baseline. cd420: CD4 at 20+/-5 weeks. cd80: CD8 at baseline. cd820: CD8 at 20+/-5 weeks.\n"
            "Analyze based on your knowledge in the subject matter, the similar historical cases, the current patient features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt_k0": (
            "You are an expert diagnostician for Cardiotocography (CTGS).\n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Feature explanation: time: time to failure or censoring. trt: treatment indicator (0 = ZDV only; 1 = ZDV + ddl, 2 = ZDV + Zal, 3 = ddl only). age: age (yrs) at baseline. wtkg: weight (kg) at baseline. hemo: hemophilia (0=no, 1=yes). homo: sexual orientation, homosexual activity (0=no, 1=yes). drugs: history of IV drug use (0=no, 1=yes). karnof: Karnofsky score (on a scale of 0-100). oprior: Non-ZDV antiretroviral therapy pre-175 (0=no, 1=yes). z30: ZDV in the 30 days prior to 175 (0=no, 1=yes). zprior: ZDV prior to 175 (0=no, 1=yes). preanti: number of days pre-175 anti-retroviral therapy. race: race (0=White, 1=non-white). gender: gender (0=F, 1=M). str2: antiretroviral history (0=naive, 1=experienced). strat: antiretroviral history stratification (1='Antiretroviral Naive',2='> 1 but <= 52 weeks of prior. symptom: symptomatic indicator (0=asymp, 1=symp). treat: treatment indicator (0=ZDV only, 1=others). offtrt: indicator of off-trt before 96+/-5 weeks (0=no,1=yes). cd40: CD4 at baseline. cd420: CD4 at 20+/-5 weeks. cd80: CD8 at baseline. cd820: CD8 at 20+/-5 weeks.\n"
            "Analyze based on your knowledge in the subject matter, the current patient features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        )
    },
    "titanic": {
        "acquisition_prompt": (
            "You are an expert in the Titanic incident. You are tasked to predict the survival of passengers based on the features. \n"
            "Current passenger features observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar Passenger profiles:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Feature explanation: PClass: Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd. Sex: Gender: male=1, female=2. Age: Age in years. SibSp: Number of siblings and spouses aboard. Parch: Number of parents and children aboard. Fare: Passenger fare. C: Embarked from Cherbourg, Q: Embarked from Queenstown, S: Embarked from Southampton."
            "Analyze based on the similar passenger profiles, the current passenger features, and your knowledge in the subject matter. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0": (
            "You are an expert in the Titanic incident. You are tasked to predict the survival of passengers based on the features. \n"
            "Current passenger features observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Feature explanation: PClass: Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd. Sex: Gender: male=1, female=2. Age: Age in years. SibSp: Number of siblings and spouses aboard. Parch: Number of parents and children aboard. Fare: Passenger fare. C: Embarked from Cherbourg, Q: Embarked from Queenstown, S: Embarked from Southampton."
            "Analyze based on your knowledge in the subject matter and the current passenger features. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt": (
            "You are an expert in the Titanic incident. You are tasked to predict the survival of passengers based on the features. \n"
            "Current passenger features observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Passenger profiles:\n{cases_str}\n"
            "Feature explanation: PClass: Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd. Sex: Gender: male=1, female=2. Age: Age in years. SibSp: Number of siblings and spouses aboard. Parch: Number of parents and children aboard. Fare: Passenger fare. C: Embarked from Cherbourg, Q: Embarked from Queenstown, S: Embarked from Southampton."
            "Analyze based on your knowledge in the subject matter, the similar passenger profiles, the current passenger features, the model prediction, and what is the most likely survival prediction (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt_k0": (
            "You are an expert in the Titanic incident. You are tasked to predict the survival of passengers based on the features. \n"
            "Current passenger features observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Feature explanation: PClass: Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd. Sex: Gender: male=1, female=2. Age: Age in years. SibSp: Number of siblings and spouses aboard. Parch: Number of parents and children aboard. Fare: Passenger fare. C: Embarked from Cherbourg, Q: Embarked from Queenstown, S: Embarked from Southampton."
            "Analyze based on your knowledge in the subject matter, the current passenger features, the model prediction, and what is the most likely survival prediction (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        )
    },
    "cps":{
        "acquisition_prompt":(
            "You are an expert in Cirrhosis Patient Survival Prediction. You are tasked to predict the survival of a patient. Label 0 = death, 1 = censored, 2 = censored due to liver transplantation\n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Feature explanation: N_Days:  number of days between registration and the earlier of death, transplantation, or study analysis time. Age: age in days. Sex: 1=female, 0=male. Bilirubin: serum bilirubin in [mg/dl]. Albumin: albumin in [gm/dl]. Platelets: platelets per cubic [ml/1000]. Prothrombin: prothrombin time in seconds [s]. Stage: histologic stage of disease (1, 2, 3, or 4)."
            "Analyze based on the current patient features, your knowledge in the subject matter, and the similar historical cases. Which SINGLE feature should be acquired next to improve diagnosis? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0":(
            "You are an expert in Cirrhosis Patient Survival Prediction. You are tasked to predict the survival of a patient. Label 0 = death, 1 = censored, 2 = censored due to liver transplantation\n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Feature explanation: N_Days:  number of days between registration and the earlier of death, transplantation, or study analysis time. Age: age in days. Sex: 1=female, 0=male. Bilirubin: serum bilirubin in [mg/dl]. Albumin: albumin in [gm/dl]. Platelets: platelets per cubic [ml/1000]. Prothrombin: prothrombin time in seconds [s]. Stage: histologic stage of disease (1, 2, 3, or 4)."
            "Analyze based on your knowledge in the subject matter and the current patient features. Which SINGLE feature should be acquired next to improve diagnosis? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt":(
            "You are an expert in Cirrhosis Patient Survival Prediction. You are tasked to predict the survival of a patient. Label 0 = death, 1 = censored, 2 = censored due to liver transplantation\n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Feature explanation: N_Days:  number of days between registration and the earlier of death, transplantation, or study analysis time. Age: age in days. Sex: 1=female, 0=male. Bilirubin: serum bilirubin in [mg/dl]. Albumin: albumin in [gm/dl]. Platelets: platelets per cubic [ml/1000]. Prothrombin: prothrombin time in seconds [s]. Stage: histologic stage of disease (1, 2, 3, or 4)."
            "Analyze based on your knowledge in the subject matter, the similar historical cases, the current patient features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt_k0":(
            "You are an expert in Cirrhosis Patient Survival Prediction. You are tasked to predict the survival of a patient. Label 0 = death, 1 = censored, 2 = censored due to liver transplantation\n"
            "Current Patient Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Feature explanation: N_Days:  number of days between registration and the earlier of death, transplantation, or study analysis time. Age: age in days. Sex: 1=female, 0=male. Bilirubin: serum bilirubin in [mg/dl]. Albumin: albumin in [gm/dl]. Platelets: platelets per cubic [ml/1000]. Prothrombin: prothrombin time in seconds [s]. Stage: histologic stage of disease (1, 2, 3, or 4)."
            "Analyze based on your knowledge in the subject matter, the current patient features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        )
    },
    "german_credit":{
            "acquisition_prompt":(
            "You are an expert in credit underwriting. You are tasked to predict the user's loan default risk. Label 0 = good, 1 = bad\n"
            "Current User Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Feature explanation: checking_status: Status of existing checking account, duration: duration in months, credit_history: credit history, purpose: purpose of the loan, credit_amount: credit amount , savings_status: saving account/bonds, employment: Present employment since, installment_commitment: Installment rate in percentage of disposable income, personal_status: Personal status and sex, other_parties: Other debtors / guarantors, residence_since: Present residence since, property_magnitude: property, age: age in years, other_payment_plans: other installment plans, housing: housing, existing_credits: Number of existing credits at this bank, job: job, num_dependents: Number of people being liable to provide maintenance for, own_telephone: own telephone, foreign_worker: foreign worker."
            "Analyze based on the current user features, your knowledge in the subject matter, and the similar historical cases. Which SINGLE feature should be acquired next to improve diagnosis? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0":(
            "You are an expert in credit underwriting. You are tasked to predict the user's loan default risk. Label 0 = good, 1 = bad\n"
            "Current User Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Feature explanation: checking_status: Status of existing checking account, duration: duration in months, credit_history: credit history, purpose: purpose of the loan, credit_amount: credit amount , savings_status: saving account/bonds, employment: Present employment since, installment_commitment: Installment rate in percentage of disposable income, personal_status: Personal status and sex, other_parties: Other debtors / guarantors, residence_since: Present residence since, property_magnitude: property, age: age in years, other_payment_plans: other installment plans, housing: housing, existing_credits: Number of existing credits at this bank, job: job, num_dependents: Number of people being liable to provide maintenance for, own_telephone: own telephone, foreign_worker: foreign worker."
            "Analyze based on your knowledge in the subject matter and the current user features. Which SINGLE feature should be acquired next to improve diagnosis? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt":(
            "You are an expert in credit underwriting. You are tasked to predict the user's loan default risk. Label 0 = good, 1 = bad\n"
            "Current User Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Feature explanation: checking_status: Status of existing checking account, duration: duration in months, credit_history: credit history, purpose: purpose of the loan, credit_amount: credit amount , savings_status: saving account/bonds, employment: Present employment since, installment_commitment: Installment rate in percentage of disposable income, personal_status: Personal status and sex, other_parties: Other debtors / guarantors, residence_since: Present residence since, property_magnitude: property, age: age in years, other_payment_plans: other installment plans, housing: housing, existing_credits: Number of existing credits at this bank, job: job, num_dependents: Number of people being liable to provide maintenance for, own_telephone: own telephone, foreign_worker: foreign worker."
            "Analyze based on your knowledge in the subject matter, the similar historical cases, the current user features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt_k0":(
            "You are an expert in credit underwriting. You are tasked to predict the user's loan default risk. Label 0 = good, 1 = bad\n"
            "Current User Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Feature explanation: checking_status: Status of existing checking account, duration: duration in months, credit_history: credit history, purpose: purpose of the loan, credit_amount: credit amount , savings_status: saving account/bonds, employment: Present employment since, installment_commitment: Installment rate in percentage of disposable income, personal_status: Personal status and sex, other_parties: Other debtors / guarantors, residence_since: Present residence since, property_magnitude: property, age: age in years, other_payment_plans: other installment plans, housing: housing, existing_credits: Number of existing credits at this bank, job: job, num_dependents: Number of people being liable to provide maintenance for, own_telephone: own telephone, foreign_worker: foreign worker."
            "Analyze based on your knowledge in the subject matter, the current user features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        )
    },
    "income":{
            "acquisition_prompt":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current adult's Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Analyze based on the current adult's features, your knowledge in the subject matter, and the similar historical cases. Which SINGLE feature should be acquired next to improve diagnosis? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Analyze based on your knowledge in the subject matter and the current adult features. Which SINGLE feature should be acquired next to improve diagnosis? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Analyze based on your knowledge in the subject matter, the similar historical cases, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt_k0":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Analyze based on your knowledge in the subject matter, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        )
    },
    "student":{
        "acquisition_prompt":(
            "You are an expert in student performance prediction. You are tasked to predict a student's performance from 0 to 3. Higher score means better performance.\n"
            "Current student's Features observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar student profiles:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Feature description: school: student's school. sex: student's sex. age: student's age (numeric: from 15 to 22). address: student's home address type. famsize: family size. Pstatus: parent's cohabitation status. Medu: mother's education. Fedu: father's education. Mjob: mother's job. Fjob: father's job. reason: reason to choose this school. guardian: student's guardian. traveltime: home to school travel time. studytime: weekly study time. failures: number of past class failures (numeric: n if 1<=n<3, else 4). schoolsup: extra educational support. famsup: family educational support. paid: extra paid classes within the course subject (Math or Portuguese). activities: extra-curricular activities. nursery: attended nursery school. higher: wants to take higher education. internet: Internet access at home. romantic: with a romantic relationship. famrel: quality of family relationships (numeric: from 1 - very bad to 5 - excellent). freetime: free time after school (numeric: from 1 - very low to 5 - very high). goout: going out with friends (numeric: from 1 - very low to 5 - very high). Dalc: workday alcohol consumption (numeric: from 1 - very low to 5 - very high). Walc: weekend alcohol consumption (numeric: from 1 - very low to 5 - very high). health: current health status (numeric: from 1 - very bad to 5 - very good). absences: number of school absences (numeric: from 0 to 93)."
            "Analyze based on the current student's features, your knowledge in the subject matter, and the similar student profiles. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0":(
            "You are an expert in student performance prediction. You are tasked to predict a student's performance from 0 to 3. Higher score means better performance.\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Feature description: school: student's school. sex: student's sex. age: student's age (numeric: from 15 to 22). address: student's home address type. famsize: family size. Pstatus: parent's cohabitation status. Medu: mother's education. Fedu: father's education. Mjob: mother's job. Fjob: father's job. reason: reason to choose this school. guardian: student's guardian. traveltime: home to school travel time. studytime: weekly study time. failures: number of past class failures (numeric: n if 1<=n<3, else 4). schoolsup: extra educational support. famsup: family educational support. paid: extra paid classes within the course subject (Math or Portuguese). activities: extra-curricular activities. nursery: attended nursery school. higher: wants to take higher education. internet: Internet access at home. romantic: with a romantic relationship. famrel: quality of family relationships (numeric: from 1 - very bad to 5 - excellent). freetime: free time after school (numeric: from 1 - very low to 5 - very high). goout: going out with friends (numeric: from 1 - very low to 5 - very high). Dalc: workday alcohol consumption (numeric: from 1 - very low to 5 - very high). Walc: weekend alcohol consumption (numeric: from 1 - very low to 5 - very high). health: current health status (numeric: from 1 - very bad to 5 - very good). absences: number of school absences (numeric: from 0 to 93)."
            "Analyze based on your knowledge in the subject matter and the current adult features. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Analyze based on your knowledge in the subject matter, the similar historical cases, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt_k0":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Analyze based on your knowledge in the subject matter, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        )
    },
    "wine":{
        "acquisition_prompt":(
            "You are an expert in wine scoring. You are tasked to predict a wine's score from 0 to 2. Higher score means better performance.\n"
            "Current wine's Features observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar wine profiles:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Analyze based on the current wine's features, your knowledge in the subject matter, and the similar wine profiles. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0":(
            "You are an expert in wine scoring. You are tasked to predict a wine's score from 0 to 2. Higher score means better performance.\n"
            "Current wine's Features observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Analyze based on the current wine's features, your knowledge in the subject matter. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Analyze based on your knowledge in the subject matter, the similar historical cases, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: Decision: Diagnosis is [Label]"
        ),
        "final_prompt_k0":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Analyze based on your knowledge in the subject matter, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: Decision: Diagnosis is [Label]"
        )
    },
    "nursery":{
        "acquisition_prompt":(
            "You are an expert in processing nursery school application. You are tasked to predict an applicant's score from 0 to 3. Higher score means more recommended to be accepted.\n"
            "Current applicant's Features observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar applicant profiles:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Feature description:  parents: Parents' occupation. has_nurs: Child's nursery. form: Form of the family. children: Number of children. housing: Housing conditions. finance: Financial standing of the family. social: Social conditions. health: Health conditions."
            "Analyze based on the current applicant's features, your knowledge in the subject matter, and the similar applicant profiles. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0":(
            "You are an expert in processing nursery school application. You are tasked to predict an applicant's score from 0 to 3. Higher score means more recommended to be accepted.\n"
            "Current applicant's Features observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Feature description:  parents: Parents' occupation. has_nurs: Child's nursery. form: Form of the family. children: Number of children. housing: Housing conditions. finance: Financial standing of the family. social: Social conditions. health: Health conditions.\n"
            "Analyze based on the current applicant's features, your knowledge in the subject matter, and the similar applicant profiles. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Analyze based on your knowledge in the subject matter, the similar historical cases, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: Decision: Diagnosis is [Label]"
        ),
        "final_prompt_k0":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Analyze based on your knowledge in the subject matter, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: Decision: Diagnosis is [Label]"
        )
    },
    "fraud":{
        "acquisition_prompt":(
            "You are an expert in mircro loan fraud detection. You are tasked to predict a loan applicantion's fraud label. Label 0 means not fraud, 1 means fraud.\n"
            "Current application's Features observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar application profiles:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Feature description: num_prior_defaults records how many previous loans the borrower has failed to repay, serving as a direct indicator of historical credit risk; identity_verification_passed indicates whether the applicant successfully passed identity and KYC checks, reflecting the credibility of the claimed identity; device_fingerprint_match captures whether the device used for the application is consistent with the borrower’s historical device usage, signaling behavioral authenticity; application_velocity_24h measures how many loan applications were submitted by the same device or network within the last 24 hours, acting as a proxy for automated or coordinated activity; income_to_loan_ratio represents the borrower’s declared monthly income relative to the requested loan amount, providing a basic affordability check; account_age_days measures how long the user account has existed, with newer accounts generally exhibiting higher uncertainty; employment_type categorizes the borrower’s employment status as salaried, self-employed, or unemployed, loosely reflecting income stability; repayment_method indicates the channel through which repayments are made, which can differ in traceability and risk exposure; geo_risk_score quantifies the historical fraud propensity of the borrower’s location based on past data; previous_loan_count counts the number of successfully completed loans and serves as evidence of repayment reliability; application_hour records the hour of day when the loan application was submitted, capturing weak temporal patterns in user behavior; mobile_os identifies the operating system of the applicant’s device, which may reflect ecosystem-level usage differences but carries little direct risk information; and marketing_campaign_id denotes the acquisition campaign through which the user was onboarded, often correlating with traffic volume rather than fraudulent intent."
            "Analyze based on the current application's features, your knowledge in the subject matter, and the similar application profiles. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0":(
            "You are an expert in mircro loan fraud detection. You are tasked to predict a loan applicantion's fraud label. Label 0 means not fraud, 1 means fraud.\n"
            "Current application's Features observed: {observed_dict}\n"
            # "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Feature description: num_prior_defaults records how many previous loans the borrower has failed to repay, serving as a direct indicator of historical credit risk; identity_verification_passed indicates whether the applicant successfully passed identity and KYC checks, reflecting the credibility of the claimed identity; device_fingerprint_match captures whether the device used for the application is consistent with the borrower’s historical device usage, signaling behavioral authenticity; application_velocity_24h measures how many loan applications were submitted by the same device or network within the last 24 hours, acting as a proxy for automated or coordinated activity; income_to_loan_ratio represents the borrower’s declared monthly income relative to the requested loan amount, providing a basic affordability check; account_age_days measures how long the user account has existed, with newer accounts generally exhibiting higher uncertainty; employment_type categorizes the borrower’s employment status as salaried, self-employed, or unemployed, loosely reflecting income stability; repayment_method indicates the channel through which repayments are made, which can differ in traceability and risk exposure; geo_risk_score quantifies the historical fraud propensity of the borrower’s location based on past data; previous_loan_count counts the number of successfully completed loans and serves as evidence of repayment reliability; application_hour records the hour of day when the loan application was submitted, capturing weak temporal patterns in user behavior; mobile_os identifies the operating system of the applicant’s device, which may reflect ecosystem-level usage differences but carries little direct risk information; and marketing_campaign_id denotes the acquisition campaign through which the user was onboarded, often correlating with traffic volume rather than fraudulent intent."
            "Analyze based on the current application's features and your knowledge in the subject matter. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Analyze based on your knowledge in the subject matter, the similar historical cases, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: Decision: Diagnosis is [Label]"
        ),
        "final_prompt_k0":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Analyze based on your knowledge in the subject matter, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: Decision: Diagnosis is [Label]"
        )
    },
    "pet":{
        "acquisition_prompt":(
            "You are an expert in pet adoption in Malaysia. You are tasked to predict how fast a pet will get adopted on a scale of 0 to 4. Lower means adopted faster.\n"
            "Current pet's Features observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar pet profiles:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Feature description: Type: Type of animal. Age: Age of pet when listed, in months. Breed1: Primary breed of pet. Breed2: Secondary breed of pet, if pet is of mixed breed. Gender: Gender of pet. Color1: Color 1 of pet. Color2: Color 2 of pet. Color3: Color 3 of pet. MaturitySize: Size at maturity. FurLength: Fur length. Vaccinated: Pet has been vaccinated. Dewormed: Pet has been dewormed. Sterilized: Pet has been spayed / neutered. Health: Health Condition. Fee: Adoption fee. State: State location in Malaysia. VideoAmt: Total uploaded videos for this pet. PhotoAmt: Total uploaded photos for this pet."
            "Analyze based on the current pet's features, your knowledge in the subject matter, and the similar pet profiles. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "acquisition_prompt_k0":(
            "You are an expert in pet adoption in Malaysia. You are tasked to predict how fast a pet will get adopted on a scale of 0 to 4. Lower means adopted faster.\n"
            "Current pet's Features observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Available Features: {available_feats_str}\n"
            "Feature description: Type: Type of animal. Age: Age of pet when listed, in months. Breed1: Primary breed of pet. Breed2: Secondary breed of pet, if pet is of mixed breed. Gender: Gender of pet. Color1: Color 1 of pet. Color2: Color 2 of pet. Color3: Color 3 of pet. MaturitySize: Size at maturity. FurLength: Fur length. Vaccinated: Pet has been vaccinated. Dewormed: Pet has been dewormed. Sterilized: Pet has been spayed / neutered. Health: Health Condition. Fee: Adoption fee. State: State location in Malaysia. VideoAmt: Total uploaded videos for this pet. PhotoAmt: Total uploaded photos for this pet."
            "Analyze based on the current pet's features, your knowledge in the subject matter. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Analyze based on your knowledge in the subject matter, the similar historical cases, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: Decision: Diagnosis is [Label]"
        ),
        "final_prompt_k0":(
            "You are an expert in income prediction. You are tasked to predict an adult's income. Label 0 means <=50k, 1 means >50k\n"
            "Current Adult Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Analyze based on your knowledge in the subject matter, the current adult features, the model prediction, and what is the most likely diagnosis (Label)? Note that the model prediction might be inaccurate.\n"
            "Return in the format: Decision: Diagnosis is [Label]"
        )
    },
    "anonymous": {
        "acquisition_prompt": (
            "You are an expert pattern matcher. You are tasked to predict the label of an instance based on the features.\n"
            "Current Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {current_pred_label} (Confidence: {current_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Available Features: {available_feats_str}\n"
            "Analyze based on the similar historical cases and the current features. Which SINGLE feature should be acquired next to improve prediction? All the features costs are equal.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        ),
        "final_prompt": (
            "You are an expert pattern matcher. You are tasked to predict the label of an instance based on the features.\n"
            "Current Features Observed: {observed_dict}\n"
            "Current Model Prediction: Label {nn_pred_label} (Confidence: {nn_confidence:.4f})\n"
            "Similar Historical Cases:\n{cases_str}\n"
            "Analyze based on the similar historical cases, the current features, the model prediction, and what is the most likely label? Note that the model prediction might be inaccurate.\n"
            "Return in the format: [Reasoning]. Decision: Acquire [feature_name] now."
        )
    }
}




