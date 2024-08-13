## To start

The folder [data/](data/) contains resulting images that was collected in accessing the fairness. For example, visit folders such as [data/zhipu/overweight/img_seq2/original/](data/zhipu/overweight/img_seq2/original/) for a sequence of images by repeatedly triggering the GenAI tool from ZHIPU, with the instruction to create overweight person while keeping image diversity. 

* [Part_A_fairness_assessment.ipynb](Part_A_fairness_assessment.ipynb) for assessing the fairness of "poor" and "successful business leader" with OpenAI
* [Part_B_ZHIPU_all_paired_fairness.ipynb](Part_B_ZHIPU_all_paired_fairness.ipynb) for assessing the all-paired-fairness of ZHIPU AI generated images subject to "successful person" (generating the tendency of coverage increase), as well as suggesting image attributes that can maximally increase fairness metrics
* [Part_C_OpenAI_GenAI_fairness_enforcement.ipynb](Part_C_OpenAI_GenAI_fairness_enforcement.ipynb) for enforcing fairness subject to "gender" or "demographics". 
* [Part_D_Zhipu_GAI_Fairness_enforcement.ipynb](Part_D_Zhipu_GAI_Fairness_enforcement.ipynb) for enforcing fairness using Zhipu tool. 

Note that some commonly used packages might need to be installed via pip. 

## Credentials
For running Part C, please create a new file called credentials.json, with contents like below.

{
        "openAI_Key": "PLEASE REPLACE IT WITH YOUR OWN KEY",
		"zhipu_Key": "PLEASE REPLACE IT WITH YOUR OWN KEY"
}


