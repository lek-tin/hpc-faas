var express = require('express');
var router = express.Router();

router.get('/hashing', function(req, res, next) {
  res.render('func/hashing', { });
});

router.get('/merge-sort', function(req, res, next) {
  res.render('func/merge-sort', { });
});

router.get('/edge-detection', function(req, res, next) {
  res.render('func/edge-detection', { });
});

module.exports = router;
