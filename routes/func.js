var express = require('express')
var router = express.Router()
var routes = require('../config/routes')
const exec = require('child_process').exec
var multer  = require('multer')


var upload_edge_detection = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, `public/rest/${routes.EDGE_DETECTION}`)
    },
    filename: (req, file, cb) => {
      cb(null, new Date().getTime() + '.png')
    }
  }),
  limits: { fileSize: 1024 * 1024 * 5 },
  fileFiler: (req, file, cb) => {
    if (file.mimetye !== 'image/png') {
      cb(null, true);
    }
    cb(new Error('Only PNG file accepted'), true);
  }
})

var upload_res_hashing = multer({ dest: `functions/rest/${routes.RSA_CRYPT}` })

router.get(`/${routes.RSA_CRYPT}`, function(req, res, next) {
  res.render(`func/${routes.RSA_CRYPT}`, { });
});

router.get(`/${routes.EDGE_DETECTION}`, function(req, res, next) {
  res.render(`func/${routes.EDGE_DETECTION}`, {});
});

router.post(`/${routes.RSA_CRYPT}`,
  upload_res_hashing.single('input'),
  function(req, res, next) {
    const job = exec(`sh functions/${routes.RSA_CRYPT}.sh`,
      function (error, stdout, stderr) {
        console.log('stdout: ' + stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
          console.log('exec error: ' + error);
        }
    });
    job.on('exit', function (code) {
      console.log('RSA Encryption/Decryption job done');
    });
    res.send('RSA Encryption respond with a POST resource');
});

router.post(`/${routes.EDGE_DETECTION}`,
  upload_edge_detection.single('inputImage'),
  function(req, res, next) {
    const dest = req.file.destination.split('/').slice(1).join('/');
    const imageId = req.file.filename.split('.')[0];
    const resultImage = `${dest}/${imageId}_gpu.png`
    const job = exec(`sh functions/${routes.EDGE_DETECTION}.sh ${imageId}`,
      function (error, stdout, stderr) {
        console.log('stdout: ' + stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
          console.log('exec error: ' + error);
        }
        res.json({
          resultImage
        });
    });
    job.on('exit', function (code) {
      console.log('edge detection job done');
    });
});

module.exports = router;
