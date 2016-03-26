'use strict';
const fs = require('fs');
const TextDataset = require('./src/textdataset');
const NNModel = require('./src/nnmodel');


//the text dataset
let td = new TextDataset();


// prediction params
var sample_softmax_temperature = 1.0; // how peaky model predictions should be

//const R = require('./src/recurrent.js');

//load the dataset
let data_sents_raw = fs.readFileSync('text_example.txt', 'utf8').split('\n');
let data_sents = [];
for (let i = 0; i < data_sents_raw.length; i++) {
  let sent = data_sents_raw[i].trim();
  if (sent.length > 0) {
    data_sents.push(sent);
  }
}
//clear the variable
data_sents_raw = [];


td.initVocab(data_sents, 1);

// model parameters
const generator = 'rnn'; // can be 'rnn' or 'lstm'
const hidden_sizes = [10, 10]; // list of sizes of hidden layers
const letter_size = 10; // size of letter embeddings

// optimization
const regc = 0.000001; // L2 regularization strength
const learning_rate = 0.01; // learning rate
const clipval = 5.0; // clip gradients at this value

console.log(generator, td.input_size, letter_size, hidden_sizes, td.output_size);
const nm = new NNModel(generator, td.input_size, letter_size, hidden_sizes, td.output_size);

let tick_iter = 0;

console.log('found ' + td.vocab.length + ' distinct characters: ' + td.vocab.join(''));
let average_tick_time = -1;

var tick = function () {

  // sample sentence fromd data

  var sent = td.random_sentence();
  var t0 = +new Date();  // log start timestamp

  // evaluate cost function on a sentence
  var cost_struct = nm.costfun(sent, td.letterToIndex);
  console.log('perplexity: ' + cost_struct.ppl + ' cost: ' + cost_struct.cost);

  // use built up graph to compute backprop (set .dw fields in mats)
  cost_struct.G.backward();
  // perform param update
  var solver_stats = nm.step(learning_rate, regc, clipval);
  //$("#gradclip").text('grad clipped ratio: ' + solver_stats.ratio_clipped)
  console.log(solver_stats);

  var t1 = +new Date();
  var tick_time = t1 - t0;
  if (average_tick_time === -1) {
    average_tick_time = tick_time;
  }
  else{
    //weighted average
    average_tick_time = average_tick_time * 0.99 + tick_time * 0.01;
  }
  console.log(learning_rate, regc, clipval);

  // evaluate now and then
  tick_iter += 1;
  if(tick_iter % 50 === 0) {
    // draw samples
    $('#samples').html('');
    for(var q=0;q<5;q++) {
      var pred = nm.predictSentence(true, sample_softmax_temperature, td.letterToIndex, td.indexToLetter);
      var pred_div = '<div class="apred">'+pred+'</div>'
      $('#samples').append(pred_div);
    }
  }
  if(tick_iter % 10 === 0) {
    // draw argmax prediction
    $('#argmax').html('');
    var pred = nm.predictSentence(false, sample_softmax_temperature, td.letterToIndex, td.indexToLetter);
    var pred_div = '<div class="apred">'+pred+'</div>'
    $('#argmax').append(pred_div);

    // keep track of perplexity
    $('#epoch').text('epoch: ' + (tick_iter/td.epoch_size).toFixed(2) + ' ['+tick_iter+' of '+ td.epoch_size +']');

    $('#ppl').text('perplexity: ' + cost_struct.ppl.toFixed(2));

    $('#ticktime').text('forw/bwd time for latest example: ' + tick_time.toFixed(1) + ' ms average: ' + average_tick_time.toFixed(2) + ' ms');

    if(tick_iter % 200 === 0) {
      var median_ppl = median(ppl_list);
      ppl_list = [];
      pplGraph.add(tick_iter, median_ppl);
      pplGraph.drawSelf(document.getElementById("pplgraph"));
    }
  }
};

for(let tt = 0 ; tt < 30; tt++){
  //console.log(tt);
  tick();
}
/*
//OLD CODE
// prediction params
var sample_softmax_temperature = 1.0; // how peaky model predictions should be

// model parameters
let generator = 'rnn'; // can be 'rnn' or 'lstm'
let hidden_sizes = [10, 10]; // list of sizes of hidden layers
let letter_size = 10; // size of letter embeddings

// optimization
let regc = 0.000001; // L2 regularization strength
let learning_rate = 0.01; // learning rate
let clipval = 5.0; // clip gradients at this value


const td = new TextDataset();
td.initVocab(data_sents, 1); // takes count threshold for characters
console.log('input size: ' + td.input_size);
console.log('output size: ' + td.output_size);


const nm = new NNModel(generator, td.input_size, letter_size, hidden_sizes, td.output_size);

var tick_iter = 0;

var average_tick_time = -1;

var tick = function () {
  // sample sentence fromd data
  var sent = td.random_sentence();
  var t0 = +new Date();  // log start timestamp

  // evaluate cost function on a sentence
  var cost_struct = nm.costfun(sent, td.letterToIndex);
  console.log('perplexity: ' + cost_struct.ppl + ' cost: ' + cost_struct.cost);

  // use built up graph to compute backprop (set .dw fields in mats)
  cost_struct.G.backward();
  // perform param update
  var solver_stats = nm.step(learning_rate, regc, clipval);
  console.log(solver_stats);
  //$("#gradclip").text('grad clipped ratio: ' + solver_stats.ratio_clipped)

  var t1 = +new Date();
  var tick_time = t1 - t0;
  if (average_tick_time === -1) {
    average_tick_time = tick_time;
  }
  else {
    //weighted average
    average_tick_time = average_tick_time * 0.99 + tick_time * 0.01;
  }
  console.log(learning_rate, regc, clipval);

  // evaluate now and then
  tick_iter += 1;
  if (tick_iter % 50 === 0) {
    // draw samples
    for(let q = 0; q < 5; q++) {
      var pred = nm.predictSentence(true, sample_softmax_temperature, td.letterToIndex, td.indexToLetter);
      console.log('generated:    ' + pred);
    }
  }
  if (tick_iter % 10 === 0) {
    //  var pred = nm.predictSentence(false, sample_softmax_temperature, td);
    // keep track of perplexity
    console.log(' --------------- ');
    console.log('epoch: ' + (tick_iter/td.epoch_size).toFixed(2) + ' ['+tick_iter+' of '+ td.epoch_size +']');
    console.log('perplexity: ' + cost_struct.ppl);
    console.log('forw/bwd time for latest example: ' + tick_time.toFixed(1) + ' ms average: ' + average_tick_time.toFixed(2) + ' ms');
  }
};


for(let tt = 0 ; tt < 30; tt++){
  //console.log(tt);
  tick();
}
*/
