# fibrosis-risk-prediction

Hepatic steatosis, or nonalcoholic fatty liver disease (NAFLD), affects a significant portion of the global population and can lead to more severe liver conditions, including hepatic fibrosis. Early and accurate risk prediction of fibrosis is crucial for timely intervention. Traditional diagnostic methods are invasive and carry risks, while imaging techniques and blood-based biomarkers have limitations in routine clinical practice.
This study presents a machine learning (ML)-based clinical decision support system (CDSS) designed to assess the risk of hepatic fibrosis in patients with NAFLD using routine laboratory tests. 
The framework is developed using extensive electronic health record data from 1,272,572 patients over a 15-year clinical history. 
It provides a robust foundation for monitoring fibrosis risk by implementing a novel screening method for NAFLD patients. This method innovatively preprocesses predictors by leveraging well-established clinical indicators, such as the hepatic steatosis index and fibrosis-4 index, alongside a minimal and selected number of routine lab tests, making it practical and cost-effective for widespread clinical use. The ML core of the proposed CDSS is represented by a reliable and transparent eXtreme Gradient Boosting model.
The study's findings indicate promising results for screening and monitoring fibrosis risk in NAFLD patients, achieving the best AUC of 92.97.

To execute the script:
- Run 'main_CDSS.py' to train and evaluate the XGB model for both Case 1 and Case 2.

'XY_case1.csv' and 'XY_case2.csv' files represent the already preprocessed data to be given as input to the XGB model:
  - Case 1: 12960 patients (11242 control, 1718 fibrosis) and 86 predictors (170 Predictors+∆)
  - Case 2: 25478 patients (22087 control, 3391 fibrosis) and 90 predictors (178 Predictors+∆)

'XY_case1.csv' and 'XY_case2.csv' files represent proprietary data and actually cannot be shared. They will soon be shared as pseudodata to execute the script.
