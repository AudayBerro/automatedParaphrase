var serializeForm = function (form) {
	var obj = {};
	var formData = new FormData(form);
	for (var key of formData.keys()) {
		obj[key] = formData.get(key);
	}
	return obj;
};

function updateTextInput(val) {
    document.getElementById('slider_value_input').value=val; 
}


document.getElementById('monselect').addEventListener('change',function(){

    var disable_pivot = false;//disable pre_trained_mt and pruning radio when Pivot-Translagtion component not in the configuration
    var disable_slider = false;//disable slider_value_input and num_seq_slider radio when T5 component not in the configuration

    // pre_trained_mt and pruning radio section
    if( ['c2','c3','c5','c12'].includes(this.value) ){
        disable_pivot = true;
    }
    
    document.querySelectorAll('#pivot1,#pivot2,#pre1,#pre2').forEach(function(e,i){
    e.disabled = disable_pivot
    })

    //slider_value_input and prnum_seq_slideruning radio section
    if( ['c1','c2','c4','c8'].includes(this.value) ){
        disable_slider = true;
    }
    
    document.querySelectorAll('#slider_value_input,#num_seq_slider').forEach(function(e,i){
    e.disabled = disable_slider
    })
    
})


//pop-up settings
// Get the modal
var modal = document.getElementById("myModal");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal(pop-up), close it
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}


// EventListener when the user press the submit(generate) button
document.addEventListener('submit',function(event){
    event.preventDefault();
    // console.log(new FormData(document.querySelector('#mainform')));//return false;
    document.querySelector('.result').classList.add('loading') ;

    //scroll to top of result
    document.querySelector('.result').scrollTo(0,0);
	fetch('http://localhost:5000/', {
        method: 'POST',
        headers: {
            'Content-type': 'application/json; charset=UTF-8'
        },
		body: JSON.stringify(serializeForm(event.target)),
	}).then(function (response) {
		if (response.ok) {
			return response.json();
		}
		return Promise.reject(response);
	}).then(function (data) {

        document.querySelector('.result').classList.remove('loading') ;
        document.getElementById('box').innerHTML='';
            for(var key in data){
                if(key != 'metric_score'){
                    for(var elem in data[key]){
                        var element = document.createElement('li');
                        element.innerHTML = data[key][elem]
                        document.getElementById('box').appendChild(element)
                    }

                }
                else{
                    document.getElementById('pop-up').innerHTML=''// remove old value
                    for(var elem in data[key]){
                        for(msg in data[key][elem]){
                            // metrics = metrics.concat(data[key][elem][msg])
                            var element = document.createElement('li');
                            element.innerHTML = data[key][elem][msg]
                            document.getElementById('pop-up').appendChild(element)
                        }
                    }
                    // display the pop-up
                    modal.style.display = "block";
                }
            }
	}).catch(function (error) {
        console.warn(error);
        document.querySelector('.result').classList.remove('loading');
        document.getElementById('box').innerHTML=error.statusText;
	});
})