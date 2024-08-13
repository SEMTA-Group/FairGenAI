from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

import numpy as np
from llama_index.llms.ollama import Ollama

from diffusers import DiffusionPipeline

import torch
import numpy as np
import random

import json
import sys

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(42)

# Image synthesis 
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

# Image interpretation
model_id = "vikhyatk/moondream2"
revision = "2024-07-23"
img_explain_model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
# Add .to("cuda") to move the model into GPU (it is too big to have it placed)

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)




app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


#llm = Ollama(model="llama3", request_timeout=120.0)
llm = Ollama(model="gemma2:2b", request_timeout=120.0)

header_description = " The person has a strong "
tail_description = " looking."

def check_prompt_topic(instruction, topic):
    # Placeholder function to call your local LLM

    resp = llm.complete("Simply answer yes or no. Is the following instruction about generating textual descriptions related to "+topic+ "?\n" + instruction)
    print("Result of checking prompt topic: "+str(resp))
    return str(resp)




def generate_description_local_llm(instruction):

    resp = llm.complete(instruction)
    print(str(resp))
    return str(resp)


def generate_image_from_text(text, output_path):
    image = pipeline(text).images[0]
    image.save(output_path)







cgvs_map = {
    'gender': ["male", "female"],
    'demographics': ["white (Caucasian) ", "black (African American)", "Indian (native American)", "Asian (e.g., Chinese, Japanese)", "Latino"]
}

image_question_map = {
    'gender':  "For the image, describe the gender of the character. If it looks like male, return \"male\". If it looks like female, return \"female\". Otherwise, return \"other\". Only return one of the three words.",
    'demographics': "For the image, describe the demograpgics / ethnicity of the character. Use one of the following options, and find the closest one: {white, black, indian, latino, asian}. If you can't really identify, then use \"other\""

}

description_annotation_map = {
    'gender': True,
    'demographics': False
}

def check_prompt_gender(instruction):
    resp = llm.complete(" Answer \"yes\" or \"no\" only. Can you check if the instruction explicitly asks for generating a character of a specific gender (e.g., male)?\n "+instruction)
    print("The prompt asks for gender: "+str(resp))
    return str(resp)

def check_prompt_demographics(instruction):
    resp = llm.complete(" Answer \"yes\", or \"no\" only. Can you check if the instruction explicitly asks for generating person of a specific demographics (e.g., white, black)?\n "+instruction)
    print("The prompt asks for demographics: "+str(resp))
    return str(resp)

query_function_map = {
    'gender': check_prompt_gender,
    'demographics': check_prompt_demographics
}

with open('config.json', 'r') as file:
    data = json.load(file)

# Change this, if you wish to enforce fairness under some other criteria (e.g., cook).
conditioned_value = data.get('fairness_enforcement', {}).get('conditioned_value')

# Change this, if you wish to enforce fairness to other dimensions
fairness_group = data.get('fairness_enforcement', {}).get('fairness_group')

# If beta = len of fairness_cgvs, then it will lead to round-robin 
beta = data.get('fairness_enforcement', {}).get('beta')

print("conditioned_value | fairness_group | beta:")
print(f"{conditioned_value} | {fairness_group} | {beta}")


fairness_cgvs = cgvs_map[fairness_group]







if beta < len(fairness_cgvs):
    err_msg = "The length of the items should at least be as large as the beta value"
    print(f"Error: {err_msg}", file=sys.stderr)
    sys.exit(1)


counters = {item: beta for item in fairness_cgvs}





@app.route('/process_instruction', methods=['POST'])
def process_instruction():

    print(counters)

    data = request.get_json()
    
    # Print the received POST data
    print("Received POST data:", data)
    
    instruction = "Generate a short text describing an imaginary example of " +data.get('instruction')+". "
    if not data.get('instruction'):
        return jsonify({'error': 'No instruction provided'}), 400

    topic_monitoring = False    

    # Check if the prompt is related to the topic, and if user has special requests on the dimension
    prompt_topic = check_prompt_topic(instruction, conditioned_value).strip().lower()
    print("The instruction sent by user shall be conditioned to fairness: "+ prompt_topic)
    if "yes" in prompt_topic:
        fairness_dimension_query = query_function_map[fairness_group](instruction).strip().lower()
        print("Fairness dimension on ("+fairness_group+") included? " + fairness_dimension_query)
        if "no" in fairness_dimension_query:
            # This means that the user has no special request on the concept group value (e.g., male for gender)
            topic_monitoring = True

    txt_enforcement_prompt = ""
    img_enforcement_prompt = ""
    enforced_cgv = ""

    if topic_monitoring == True:
        # Consider if enforcement should be triggered


        # Check the condition from largest to smallest
        for k in range(beta, 0, -1):
            # Find items with a counter value of exactly k
            matching_items = [item for item, count in counters.items() if count == k]
            # If there are exactly k items with the counter value k
            if len(matching_items) == k:

                # Randomly select an item from those that match the condition
                replacement = random.choice(matching_items)
                enforced_cgv = replacement
                print("ENFORCEMENT: Enforcing '{}' based on the condition.".format(replacement))

                if description_annotation_map[fairness_group] == True:
                    txt_enforcement_prompt = header_description+replacement+tail_description 
                
                img_enforcement_prompt = header_description+replacement+tail_description             
                break
    else:
        print("The topic shall not be monitored")

    # Generate image description
    llm_response = generate_description_local_llm( txt_enforcement_prompt + instruction + txt_enforcement_prompt)
    
    # Generate an image using Stable Diffusion
    image_filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join('generated_images', image_filename)
    print(str(image_filename))
    generate_image_from_text(img_enforcement_prompt + llm_response + img_enforcement_prompt, image_path)

    # Perform fainess counter update
    if topic_monitoring == True:

        concept_group_value = ""

        if enforced_cgv != "":
            # Here we are explicitly assuming that the enforcement works so there is no need to double check. 
            concept_group_value = enforced_cgv
        else:
            # Use the vision-based interpreter to check the class
            image_pil = Image.open(image_path)
            enc_image = img_explain_model.encode_image(image_pil)
            attribute = img_explain_model.answer_question(enc_image, image_question_map[fairness_group], tokenizer).strip().lower()
            print("The synthesized image has the attribute: "+attribute)
            is_contained = False
            for item in cgvs_map[fairness_group]:
                # Check if the function's output is a substring of the list item
                if attribute.lower() in item.lower():
                    concept_group_value = item
                    is_contained = True
                    break
            if is_contained == False and ("other" not in attribute):
                err_msg = "The image explaination model produces an answer (apart from other): "+ attribute
                print(f"Error: {err_msg}", file=sys.stderr)
                sys.exit(1)

        for item in counters:
            if item == concept_group_value:
                counters[item] = beta
            else:
                counters[item] -= 1

    if enforced_cgv == "":
         return jsonify({
        'response': llm_response,
        'image_url': f"http://localhost:5000/get_image/{image_filename}"
        })
    else:
         return jsonify({
        'response': llm_response + "\n **** Fairness enforcement enabled: "+enforced_cgv +" ****\n",
        'image_url': f"http://localhost:5000/get_image/{image_filename}"
        })


@app.route('/get_image/<filename>', methods=['GET'])
def get_image(filename):
    image_path = os.path.join('generated_images', filename)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    app.run(debug=True, port=5000)