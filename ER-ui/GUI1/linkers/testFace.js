function test_Face(){
  let {PythonShell} = require('python-shell')
  let path = require('path');


  let Image = document.getElementById("Image").files[0].path;

  let options = {

    scriptPath : path.join(__dirname, '/../Engine'),
    pythonPath : 'C:/ProgramData/Anaconda3/python',
    args: [Image]
  };

let pyshell = new PythonShell('TestImage.py', options);
  pyshell.on('message', function(message){

    swal(message)
  });

  pyshell.end(function (err,code,signal){
    if (err) throw err;
    console.log('The exit code was: ' + code);
    console.log('The exit signal was: ' + signal);
    console.log('finished');
    console.log('finished');
  })

}
