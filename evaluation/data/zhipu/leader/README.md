This repository contains code to generate images of successful persons with diverse attributes, including race, gender, occupation, and body shape. The generation process is guided by prompts specifying the desired attributes for each image. Below are the instructions for generating diverse images:

Baseline (img_seq1) Prompt:

1. Input: "Please generate an image of a successful person."
   
2. Input: "Please generate another image of a successful person, different from the previous one."
   
3. Repeat: "Please generate yet another image of a successful person, different from previous ones."

In order to increase the pair-wise coverage, we use detailed prompt for generating image sequences 3 to 7 as follows:

1. Input: "Please generate an image of a successful person. Please state the attributes of the generated image in each dimension of demographic (race), gender, occupation, and body shape after generation."
   
2. Input: "Please generate another image of a successful person, different from the previous one in terms of one of the following dimensions: demographic (race), occupation, gender, and body shape. Please state the attributes of the generated image in each dimension of demographic (race), gender, occupation, and body shape after generation."
   
3. Repeat until 80 images are generated: "Please generate yet another image of a successful person, different from previous ones in terms of one of the following dimensions: demographic (race), occupation, gender, and body shape. Please state the attributes of the generated image in each dimension of demographic (race), gender, occupation, and body shape after generation."