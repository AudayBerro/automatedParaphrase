function displayResult(){
    setTimeout(addParaphrases,3000);
}

function addParaphrases(){
    var a = ['will contain generated paraphrases']
        document.getElementById('box').innerHTML=''
    for(var i=0;i<a.length;i++ ){
        var element = document.createElement('li');
        element.innerHTML=a[i]
        document.getElementById('box').appendChild(element)
    }
}

document.getElementById('monselect').addEventListener('change',function(){
    var disable = false;
    if(this.value=='valeur2'){
        disable = true;
    }
    
    document.querySelectorAll('#pivot1,#pivot2,#pre1,#pre2').forEach(function(e,i){
    e.disabled = disable
    })
    
})