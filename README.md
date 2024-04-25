# GenAI-fairness-monitor
Fairness specification, assessment, and enforcement for generative AI

## To start

* [Part_A_fairness_assessment.ipynb](Part_A_fairness_assessment.ipynb) for assessing the fairness of "poor" and "successful business leader" with OpenAI
* [Part_B_ZHIPU_all_paired_fairness.ipynb](Part_B_ZHIPU_all_paired_fairness.ipynb) for assessing the all-paired-fairness of ZHIPU AI generated images subject to "successful person" (generating the tendency of coverage increase), as well as suggesting image attributes that can maximally increase fairness metrics
* [Part_C_OpenAI_GenAI_fairness_enforcement.ipynb](Part_C_OpenAI_GenAI_fairness_enforcement.ipynb) for enforcing fairness subject to "gender" or "demographics". 

## Credentials
For running Part C, please create a new file called credentials.json, with contents like below.

{
        "openAI_Key": "PLEASE REPLACE IT WITH YOUR OWN KEY",
		"zhipu_Key": "PLEASE REPLACE IT WITH YOUR OWN KEY"
}


