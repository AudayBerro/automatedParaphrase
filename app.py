from flask import Flask,render_template,request,jsonify
import main
""" pipeline FLask application """

app = Flask(__name__)

@app.route('/',methods=['post','get'])
def index():
    if request.method == 'POST':
        # get user pipeline selected configuration option
        data = request.get_json(force=True)
        # config = request.form.get('configuration'
       
        config = data['configuration']#return string
        
        # get user sentence
        sentence = data['user_utterance']

        # check if pivot-level radio option is not disabled
        if data['pivot_level']:
            # get selected pivot level option: 1-pivot or 2-pivot
            pivot_level = data['pivot_level']#return string

            # Machine Translator option: pre-trained MT(e.g. Huggingface Marian MT) or Online MT model(Deepl,Google)
            pre_trained_mt = data['pre_trained_mt']#return string
            paraphrases = main.generate_from_gui(sentence,config,pivot_level,pre_trained_mt)
        else:
            paraphrases = main.generate_from_gui(sentence,config)
        return jsonify(paraphrases)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)