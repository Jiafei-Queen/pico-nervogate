const http = require('http');

const chat = {
  id: 'chatcmpl_mock', object: 'chat.completion', created: 1, model: 'mock',
  choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 },
};
const anthropic = {
  id: 'msg_mock', type: 'message', role: 'assistant', model: 'mock',
  content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', stop_sequence: null,
  usage: { input_tokens: 4, output_tokens: 1 },
};
const responses = {
  id: 'resp_mock', object: 'response', created_at: 1, model: 'mock', status: 'completed',
  output: [{ id: 'msg_mock', type: 'message', role: 'assistant', status: 'completed',
    content: [{ type: 'output_text', text: 'ok', annotations: [] }] }],
  usage: { input_tokens: 4, output_tokens: 1, total_tokens: 5 },
};
const chunk = (content, finishReason = null) => ({
  id: 'chatcmpl_mock', object: 'chat.completion.chunk', created: 1, model: 'mock',
  choices: [{ index: 0, delta: content === null ? {} : { content }, finish_reason: finishReason }],
});
const chatStream = [
  `data: ${JSON.stringify(chunk('ok'))}\n\n`,
  `data: ${JSON.stringify(chunk(null, 'stop'))}\n\n`,
  'data: [DONE]\n\n',
].join('');
const anthropicStream = [
  ['message_start', { type: 'message_start', message: { id: 'msg_mock', type: 'message', role: 'assistant', model: 'mock', content: [], usage: { input_tokens: 4, output_tokens: 0 } } }],
  ['content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }],
  ['content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'ok' } }],
  ['content_block_stop', { type: 'content_block_stop', index: 0 }],
  ['message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 1 } }],
  ['message_stop', { type: 'message_stop' }],
].map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join('');
const responseEvents = [
  { type: 'response.created', response: { id: 'resp_mock', object: 'response', created_at: 1, model: 'mock', status: 'in_progress', output: [] } },
  { type: 'response.output_item.added', output_index: 0, item: { id: 'msg_mock', type: 'message', role: 'assistant', status: 'in_progress', content: [] } },
  { type: 'response.content_part.added', item_id: 'msg_mock', output_index: 0, content_index: 0, part: { type: 'output_text', text: '', annotations: [] } },
  { type: 'response.output_text.delta', item_id: 'msg_mock', output_index: 0, content_index: 0, delta: 'ok' },
  { type: 'response.output_text.done', item_id: 'msg_mock', output_index: 0, content_index: 0, text: 'ok' },
  { type: 'response.content_part.done', item_id: 'msg_mock', output_index: 0, content_index: 0, part: { type: 'output_text', text: 'ok', annotations: [] } },
  { type: 'response.output_item.done', output_index: 0, item: responses.output[0] },
  { type: 'response.completed', response: responses },
].map(data => `data: ${JSON.stringify(data)}\n\n`).join('');

const server = http.createServer((req, res) => {
  const chunks = [];
  req.on('data', chunk => chunks.push(chunk));
  req.on('end', () => {
    let body = {};
    try { body = JSON.parse(Buffer.concat(chunks).toString()); } catch {}
    const protocol = req.url.endsWith('/messages') ? 'anthropic'
      : req.url.endsWith('/responses') ? 'responses' : 'chat';
    res.setHeader('content-type', body.stream ? 'text/event-stream' : 'application/json');
    res.writeHead(200);
    if (body.stream) {
      res.end({ chat: chatStream, anthropic: anthropicStream, responses: responseEvents }[protocol]);
    } else {
      res.end(JSON.stringify({ chat, anthropic, responses }[protocol]));
    }
  });
});

server.listen(Number(process.env.MOCK_PORT), '127.0.0.1');
