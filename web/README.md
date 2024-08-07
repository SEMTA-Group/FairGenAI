## Simple GenAI web service that enforces worst-case fairness

### A. Limitations

Currently, we implement "gender fairness" conditional to "successful business leader". Therefore, if the user demands something else, enforcement will not be triggered. In addition, when the user explicitly demands a female, then enforcement and counting will not be triggered. 

### B. Execute the Application


1. Start the Flask Server: Open a terminal, navigate to the directory containing server.py, and run (or run from VS Code)


    $ python server.py

2. Start the LiveServer: Open VS Code and start the Live Server to serve your index.html.

    * Important: Disable LiveServer automatic reload of the server upon file change [(link)](https://stackoverflow.com/questions/77120592/how-can-i-prevent-page-reloads-upon-saving-changes-with-the-live-server-extensio), as when the image is generated and stored, without disabling, the reloading will simply erase the display. 

3. Access the Application: Open your browser and navigate to the URL provided by the Live Server (e.g., http://127.0.0.1:5500/index.html).

4. Test the Application: Each time you submit the form, the application should call the LLM, generate an image using Stable Diffusion, and display both the text response and the generated image as a link.

    * In the form, please enter "successful business leader" or of similar kind, as this is the only item now being enforced for fairness. One can change it to others such as "bad cook".  

5. When fairness is enabled, it will explitly notify the user that fairness constraints enforce the generation to be "male" or "female" (for gender fairness), and inform user that they can make the gender explicit, in order not to be enforced. 

### C. Modification

One shall modify the server.py for the condition where fairness shall be enforced, as well as the worst case bound (beta).