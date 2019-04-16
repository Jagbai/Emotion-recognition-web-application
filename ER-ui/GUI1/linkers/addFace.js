function add_Face(){
  let {PythonShell} = require('python-shell')
  let path = require('path');



  let Emotion = document.getElementById("emotion").value;
  let Image = document.getElementById("Image").files[0].path;

  let options = {

    scriptPath : path.join(__dirname, '/../Engine'),
    pythonPath : 'C:/ProgramData/Anaconda3/python',
    args: [Image, Emotion]
  };

PythonShell.run('addFace.py', options, function(err, results){
  if (err) throw err;
  // results is an array consisting of messages collected during execution
   console.log('results: %j', results);
   swal("Face added!", "Congratulations! you're now part of our data", "success")
})

}
