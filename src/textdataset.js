const TextDataset = function(sents){
  this.epoch_size = -1;
  this.input_size = -1;
  this.output_size = -1;
  this.letterToIndex = {};
  this.indexToLetter = {};
  this.vocab = [];

  this.sents = [];
  /**
  * Load an array of strings as the text dataset
  */
  this.initVocab = function (sents, count_threshold) {
    this.sents = sents;
    // go over all characters and keep track of all unique ones seen
    let txt = sents.join(''); // concat all

    // count up all characters
    var d = {};
    for (var i = 0, n = txt.length; i < n; i++) {
      var txti = txt[i];
      if (txti in d) {
        d[txti]+=1;
      }
      else {
        d[txti] = 1;
      }
    }

    // NOTE: start at one because we will have START and END tokens!
    // that is, START token will be index 0 in model letter vectors
    // and END token will be index 0 in the next character softmax
    let q = 1;
    for (let ch in d) {
      if (d.hasOwnProperty(ch)) {
        if (d[ch] >= count_threshold) {
          // add character to vocab
          this.letterToIndex[ch] = q;
          this.indexToLetter[q] = ch;
          this.vocab.push(ch);
          q++;
        }
      }
    }
    this.input_size = this.vocab.length + 1;
    this.output_size = this.vocab.length + 1;
    this.epoch_size = sents.length;
  };

  this.random_sentence = function () {
    const sentix = R.randi(0, this.sents.length);
    return this.sents[sentix];
  };
};
