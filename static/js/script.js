var serializeForm = function (form) {
	var obj = {};
	var formData = new FormData(form);
	for (var key of formData.keys()) {
		obj[key] = formData.get(key);
	}
	return obj;
};

document.getElementById('monselect').addEventListener('change',function(){
    var disable = false;
    if( ['c2','c3','c5','c12'].includes(this.value) ){
        disable = true;
    }
    
    document.querySelectorAll('#pivot1,#pivot2,#pre1,#pre2').forEach(function(e,i){
    e.disabled = disable
    })
    
})

document.addEventListener('submit',function(event){
    event.preventDefault();
// console.log(new FormData(document.querySelector('#mainform')));//return false;
    document.querySelector('.result').classList.add('loading') ;
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
                for(var elem in data[key]){
                    var element = document.createElement('li');
                    element.innerHTML = data[key][elem]
                    document.getElementById('box').appendChild(element)
                }
            }
        
	}).catch(function (error) {
		console.warn(error);
	});
})