function Face_recog(){
  document.getElementById("recog").value = "Hang on..."
  let {PythonShell} = require('python-shell')
  let path = require('path');



  let options = {

    scriptPath : path.join(__dirname, '/../Engine'),
    pythonPath : 'C:/ProgramData/Anaconda3/python',

  };

let pyshell = new PythonShell('ModelPrediction.py', options);

  pyshell.end(function (err,code,signal){
    document.getElementById("recog").value = "GO"
    if (err) throw err;
    console.log('The exit code was: ' + code);
    console.log('The exit signal was: ' + signal);
    console.log('finished');
    console.log('finished');
  })

}
