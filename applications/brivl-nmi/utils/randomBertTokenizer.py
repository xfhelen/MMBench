import torch

class RandomBertTokenizer:
    def __init__(self, vocab_size=30522):
        # 创建一个随机词汇表
        self.vocab = {f"token_{i}": i for i in range(vocab_size)}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}  # 反向词汇表用于解码
        self.vocab_size = vocab_size

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors=None):
        # 简单的分词逻辑，将每个字符映射到一个随机的 token
        encoded_inputs = []
        for text in texts:
            encoded_input = [self.vocab.get(f"token_{ord(char) % self.vocab_size}", 0) for char in text]
            encoded_inputs.append(encoded_input)

        # 填充和截断
        if max_length:
            max_length = min(max_length, max(len(seq) for seq in encoded_inputs))  # 保证max_length不超过文本最大长度

        # 填充到max_length
        if padding:
            encoded_inputs = [seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length] for seq in encoded_inputs]

        # 截断
        if truncation:
            encoded_inputs = [seq[:max_length] for seq in encoded_inputs]

        # 返回张量
        encoded_tensor = torch.tensor(encoded_inputs)

        # 处理return_tensors参数，返回PyTorch张量
        if return_tensors == 'pt':
            return encoded_tensor
        return encoded_tensor.tolist()  # 如果不需要返回张量，返回list格式

    def encode(self, texts, padding=True, truncation=True, max_length=None, return_tensors='pt'):
        """将文本编码为token ID列表，兼容BERT的接口"""
        return self(texts, padding=padding, truncation=truncation, max_length=max_length, return_tensors=return_tensors)

    def decode(self, token_ids, skip_special_tokens=False):
        """将token ID解码为文本，兼容BERT的接口"""
        decoded_texts = []
        for token_id_list in token_ids:
            decoded_text = ''.join([self.reverse_vocab.get(token_id.item(), '<unk>') for token_id in token_id_list])
            decoded_texts.append(decoded_text)
        return decoded_texts

    def encode_plus(self, text, padding=True, truncation=True, max_length=None, return_tensors='pt'):
        """兼容BERT的encode_plus方法，返回token IDs和attention mask"""
        encoding = self.encode([text], padding=padding, truncation=truncation, max_length=max_length, return_tensors=return_tensors)
        attention_mask = torch.ones(encoding.shape, dtype=torch.long)  # 假设所有的token都有attention
        return {
            'input_ids': encoding,
            'attention_mask': attention_mask
        }