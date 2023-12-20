def enumerate_spans(batch):
  
  enumerated_spans_batch = []
  
  for idx in range(len(batch)):
    sentence_length = batch[idx]
    enumerated_spans = []
    for x in range(len(sentence_length)):
      for y in range(x, len(sentence_length)):
        enumerated_spans.append([x,y])
        
    enumerated_spans_batch.append(enumerated_spans)
  
  return enumerated_spans_batch
