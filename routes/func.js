var express = require('express')
var router = express.Router()
var routes = require('../config/routes')
const exec = require('child_process').exec
var multer  = require('multer')
var upload_edge_detection = multer({ dest: `functions/${routes.EDGE_DETECTION}/output` })
var upload_res_hashing = multer({ dest: `functions/${routes.RSA_CRYPT}/output` })

router.get(`/${routes.RSA_CRYPT}`, function(req, res, next) {
  res.render(`func/${routes.RSA_CRYPT}`, { });
});

router.get(`/${routes.EDGE_DETECTION}`, function(req, res, next) {
  res.render(`func/${routes.EDGE_DETECTION}`, {});
});

router.post(`/${routes.RSA_CRYPT}`,
upload_res_hashing.single(new Date().getTime()),
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
  upload_edge_detection.single(new Date().getTime()),
  function(req, res, next) {
    const job = exec(`sh functions/${routes.EDGE_DETECTION}.sh`,
      function (error, stdout, stderr) {
        console.log('stdout: ' + stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
          console.log('exec error: ' + error);
        }
    });
    job.on('exit', function (code) {
      console.log('edge detection job done');
    });
    res.send('edge detection respond with a POST resource');
});

module.exports = router;
