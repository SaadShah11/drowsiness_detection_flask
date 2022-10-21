
var drowsiness_alert_div = document.getElementById("drowsiness_alert_div")

var requestOptions = {
  method: 'GET',
  redirect: 'follow'
};

while( true ){
    fetch("http://127.0.0.1:5000/fetch_drowsiness_alert", requestOptions)
      .then(response => response.text())
      .then(result => {
        console.log("result: "+result)
        if(result) {
            drowsiness_alert_div.style.display = "block"
        } else {
            drowsiness_alert_div.style.display = "none"
        }
      })
      .catch(error => console.log('error', error));
}
