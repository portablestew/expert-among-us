import unittest
from unittest.mock import patch, Mock
from expert_among_us.embeddings.bedrock import BedrockEmbedder

class TestBedrockEmbedder(unittest.TestCase):
    @patch('src.expert_among_us.embeddings.bedrock.boto3.client')
    def test_embed(self, mock_boto3_client):
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock response body with read() method
        mock_body = Mock()
        mock_body.read.return_value = '{"embedding": ' + str([0.1] * 1024) + '}'
        mock_client.invoke_model.return_value = {
            'body': mock_body
        }
        
        model_id = "amazon.titan-embed-text-v2:0"
        embedder = BedrockEmbedder(model_id)
        
        text = "Sample text for embedding"
        embedding = embedder.embed(text)
        
        self.assertEqual(len(embedding), 1024)
        self.assertTrue(mock_client.invoke_model.called)
        
    @patch('src.expert_among_us.embeddings.bedrock.boto3.client')
    def test_embed_batch(self, mock_boto3_client):
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock response bodies with read() method
        mock_body1 = Mock()
        mock_body1.read.return_value = '{"embedding": ' + str([0.1] * 1024) + '}'
        mock_body2 = Mock()
        mock_body2.read.return_value = '{"embedding": ' + str([0.2] * 1024) + '}'
        
        mock_client.invoke_model.side_effect = [
            {'body': mock_body1},
            {'body': mock_body2}
        ]
        
        model_id = "amazon.titan-embed-text-v2:0"
        embedder = BedrockEmbedder(model_id)
        
        texts = ["Text 1", "Text 2"]
        embeddings = embedder.embed_batch(texts)
        
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 1024)
        self.assertEqual(len(embeddings[1]), 1024)
        
    @patch('src.expert_among_us.embeddings.bedrock.boto3.client')
    def test_dimension(self, mock_boto3_client):
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        model_id = "amazon.titan-embed-text-v2:0"
        embedder = BedrockEmbedder(model_id)
        
        self.assertEqual(embedder.dimension, 1024)

if __name__ == '__main__':
    unittest.main()