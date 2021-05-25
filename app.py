from flask import Flask,render_template,request,jsonify
import main
""" pipeline FLask application """

app = Flask(__name__)

@app.route('/',methods=['post','get'])
def index():
    if request.method == 'POST':
        
        data = request.get_json(force=True)

        # get user pipeline selected configuration option
        config = data['configuration']#return string
        
        # get user sentence
        sentence = data['user_utterance']

        # get candidate selection(pruning) flag
        if "pruning" in data:
            pruning = data['pruning']
        else:
            pruning = "Off"
        
        # check if pivot-level radio option is not disabled
        if "pivot_level" in data:
            # get selected pivot level option: 1-pivot or 2-pivot
            pivot_level = data['pivot_level']#return string

            # Machine Translator option: pre-trained MT(e.g. Huggingface Marian MT) or Online MT model(Deepl,Google)
            pre_trained = data['pre_trained_mt']#return string
        else:
            pivot_level = None
            pre_trained = None
        
        # check T5 num_seq_slider(number of independently computed returned sequences for each element in the batch)
        if 'num_seq_slider' in data:
            num_seq = int(data['num_seq_slider'])
        else:
            num_seq = None
        
        paraphrases = main.generate_from_gui(sentence,config,pruning=pruning,pivot_level=pivot_level,pre_trained=pre_trained,num_seq=num_seq)

        return jsonify(paraphrases)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)