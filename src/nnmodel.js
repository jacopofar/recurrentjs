const NNModel = function(generator, input_size, letter_size, hidden_sizes, output_size){
  this.solver = new R.Solver(); // should be class because it needs memory for step caches
  this.model = {};

  this.model['Wil'] = new R.RandMat(input_size, letter_size, 0, 0.08);
  const utilAddToModel = function (modelto, modelfrom) {
    for (var k in modelfrom) {
      if (modelfrom.hasOwnProperty(k)) {
        // copy over the pointer but change the key to use the append
        modelto[k] = modelfrom[k];
      }
    }
  }
  if (generator === 'rnn') {
    let rnn = R.initRNN(letter_size, hidden_sizes, output_size);
    utilAddToModel(this.model, rnn);
  } else {
    var lstm = R.initLSTM(letter_size, hidden_sizes, output_size);
    utilAddToModel(this.model, lstm);
  }

  this.to_state_object = function () {
    let out = {};
    out['hidden_sizes'] = hidden_sizes;
    out['generator'] = generator;
    out['letter_size'] = letter_size;
    let model_out = {};
    for (let k in this.model) {
      if (this.model.hasOwnProperty(k)) {
        model_out[k] = this.model[k].toJSON();
      }
    }
    out['model'] = model_out;
    var solver_out = {};
    solver_out['decay_rate'] = this.solver.decay_rate;
    solver_out['smooth_eps'] = this.solver.smooth_eps;
    let step_cache_out = {};
    for (var k in this.solver.step_cache) {
      if (this.solver.step_cache.hasOwnProperty(k)) {
        step_cache_out[k] = this.solver.step_cache[k].toJSON();
      }
    }
    solver_out['step_cache'] = step_cache_out;
    out['solver'] = solver_out;
    return out;
  };

  this.from_state_object = function (obj) {
    this.model = {};
    for (let k in obj.model) {
      if (obj.model.hasOwnProperty(k)) {
        let matjson = obj.model[k];
        this.model[k] = new R.Mat(1, 1);
        this.model[k].fromJSON(matjson);
      }
    }
    // have to reinit the solver since model changed
    this.solver = new R.Solver();
    this.solver.decay_rate = obj.solver.decay_rate;
    this.solver.smooth_eps = obj.solver.smooth_eps;
    this.solver.step_cache = {};
    for (var k in obj.solver.step_cache) {
      if (obj.solver.step_cache.hasOwnProperty(k)) {
        var matjson = obj.solver.step_cache[k];
        this.solver.step_cache[k] = new R.Mat(1, 1);
        this.solver.step_cache[k].fromJSON(matjson);
      }
    }
  };

  this.forwardIndex = function (G, ix, prev) {
    let x = G.rowPluck(this.model['Wil'], ix);
    // forward prop the sequence learner
    if (generator === 'rnn') {
      return R.forwardRNN(G, this.model, hidden_sizes, x, prev);
    } else {
      return R.forwardLSTM(G, this.model, hidden_sizes, x, prev);
    }
  };

  this.predictSentence = function (samplei = false, temperature = 1.0, td) {
    var G = new R.Graph(false);
    var s = '';
    var prev = {};
    //generate sentences of at most 160 characters
    for (let c = 0; c < 160; c++) {
      // RNN tick
      let ix = s.length === 0 ? 0 : td.letterToIndex[s[s.length - 1]];
      var lh = this.forwardIndex(G, ix, prev);
      prev = lh;

      // sample predicted letter
      let logprobs = lh.o;
      if (temperature !== 1.0 && samplei) {
        // scale log probabilities by temperature and renormalize
        // if temperature is high, logprobs will go towards zero
        // and the softmax outputs will be more diffuse. if temperature is
        // very low, the softmax outputs will be more peaky
        for (let q = 0, nq = logprobs.w.length; q < nq; q++) {
          logprobs.w[q] /= temperature;
        }
      }

      const probs = R.softmax(logprobs);
      if (samplei) {
        ix = R.samplei(probs.w);
      } else {
        ix = R.maxi(probs.w);
      }

      if (ix === 0) return s; // END token predicted, break out
      s += td.indexToLetter[ix];
    }
    return s;
  };

  this.step = function (learning_rate, regc, clipval) {
    return this.solver.step(this.model, learning_rate, regc, clipval);
  };
};
