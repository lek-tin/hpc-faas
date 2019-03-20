var express = require('express')
var router = express.Router()
var routes = require('../config/routes')
const exec = require('child_process').exec
var multer  = require('multer')
var upload_edge_detection = multer({ dest: `functions/${routes.EDGE_DETECTION}/images` })
var upload_res_hashing = multer({ dest: `functions/${routes.RSA_HASHING}/images` })

router.get(`/${routes.RSA_HASHING}`, function(req, res, next) {
  res.render(`func/${routes.RSA_HASHING}`, { });
});

router.get(`/${routes.EDGE_DETECTION}`, function(req, res, next) {
  res.render(`func/${routes.EDGE_DETECTION}`, {});
});

router.post(`/${routes.EDGE_DETECTION}`,
  upload_edge_detection.single(new Date().getTime()),
  function(req, res, next) {
    console.log(req.file);
    exec(`sh functions/${routes.EDGE_DETECTION}.sh`,
      function (error, stdout, stderr) {
        console.log('stdout: ' + stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
          console.log('exec error: ' + error);
        }
    });
    res.send('respond with a POST resource');
});

module.exports = router;
