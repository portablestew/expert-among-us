import unittest
from unittest.mock import patch, Mock
from expert_among_us.embeddings.bedrock import BedrockEmbedder

class TestBedrockEmbedder(unittest.TestCase):
    @patch('expert_among_us.utils.aws.boto3')
    def test_embed(self, mock_boto3):
        # Mock session and clients
        mock_session = Mock()
        mock_boto3.Session.return_value = mock_session
        
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {'UserId': 'test-user'}
        
        mock_bedrock_client = Mock()
        
        # Mock credentials to have no expiry (static credentials)
        mock_credentials = Mock()
        mock_credentials._expiry_time = None
        mock_session.get_credentials.return_value = mock_credentials
        
        # Configure session.client() to return appropriate clients
        def client_factory(service_name):
            if service_name == 'sts':
                return mock_sts_client
            elif service_name == 'bedrock-runtime':
                return mock_bedrock_client
            return Mock()
        
        mock_session.client.side_effect = client_factory
        
        # Mock response body with read() method
        mock_body = Mock()
        mock_body.read.return_value = '{"embedding": ' + str([0.1] * 1024) + '}'
        mock_bedrock_client.invoke_model.return_value = {
            'body': mock_body
        }
        
        model_id = "amazon.titan-embed-text-v2:0"
        embedder = BedrockEmbedder(model_id)
        
        text = "Sample text for embedding"
        embedding = embedder.embed(text)
        
        self.assertEqual(len(embedding), 1024)
        self.assertTrue(mock_bedrock_client.invoke_model.called)
        self.assertTrue(mock_sts_client.get_caller_identity.called)
        
    @patch('expert_among_us.utils.aws.boto3')
    def test_embed_batch(self, mock_boto3):
        # Mock session and clients
        mock_session = Mock()
        mock_boto3.Session.return_value = mock_session
        
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {'UserId': 'test-user'}
        
        mock_bedrock_client = Mock()
        
        # Mock credentials to have no expiry (static credentials)
        mock_credentials = Mock()
        mock_credentials._expiry_time = None
        mock_session.get_credentials.return_value = mock_credentials
        
        # Configure session.client() to return appropriate clients
        def client_factory(service_name):
            if service_name == 'sts':
                return mock_sts_client
            elif service_name == 'bedrock-runtime':
                return mock_bedrock_client
            return Mock()
        
        mock_session.client.side_effect = client_factory
        
        # Mock response bodies with read() method
        mock_body1 = Mock()
        mock_body1.read.return_value = '{"embedding": ' + str([0.1] * 1024) + '}'
        mock_body2 = Mock()
        mock_body2.read.return_value = '{"embedding": ' + str([0.2] * 1024) + '}'
        
        mock_bedrock_client.invoke_model.side_effect = [
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
        
    @patch('expert_among_us.utils.aws.boto3')
    def test_dimension(self, mock_boto3):
        # Mock session and clients
        mock_session = Mock()
        mock_boto3.Session.return_value = mock_session
        
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {'UserId': 'test-user'}
        
        mock_bedrock_client = Mock()
        
        # Mock credentials to have no expiry (static credentials)
        mock_credentials = Mock()
        mock_credentials._expiry_time = None
        mock_session.get_credentials.return_value = mock_credentials
        
        # Configure session.client() to return appropriate clients
        def client_factory(service_name):
            if service_name == 'sts':
                return mock_sts_client
            elif service_name == 'bedrock-runtime':
                return mock_bedrock_client
            return Mock()
        
        mock_session.client.side_effect = client_factory
        
        model_id = "amazon.titan-embed-text-v2:0"
        embedder = BedrockEmbedder(model_id)
        
        self.assertEqual(embedder.dimension, 1024)

if __name__ == '__main__':
    unittest.main()