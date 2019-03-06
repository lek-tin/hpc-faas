var express = require('express');
var router = express.Router();

/* GET functions listing. */
router.get('/', function(req, res, next) {
  res.send('respond with a GET resource');
});

/* POST functions listing. */
router.post('/', function(req, res, next) {
  res.send('respond with a POST resource');
});

module.exports = router;
