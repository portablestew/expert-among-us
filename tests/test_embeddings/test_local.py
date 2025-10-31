import unittest
from unittest.mock import patch, Mock, MagicMock
from expert_among_us.embeddings.local import JinaCodeEmbedder
import numpy as np


class TestJinaCodeEmbedder(unittest.TestCase):
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.compile', side_effect=lambda x: x)
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed(self, mock_sentence_transformer_class, mock_compile, mock_cuda):
        """Test basic embedding generation."""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer_class.return_value = mock_model
        
        # Mock encode to return numpy array that will be truncated to 512
        full_embedding = np.array([0.1] * 896)
        mock_model.encode.return_value = full_embedding
        
        # Initialize embedder
        model_id = "jinaai/jina-code-embeddings-0.5b"
        embedder = JinaCodeEmbedder(model_id=model_id, dimension=512)
        
        # Test embedding
        text = "Sample code for embedding"
        embedding = embedder.embed(text)
        
        # Verify results
        self.assertEqual(len(embedding), 512)
        self.assertTrue(mock_model.encode.called)
        
        # Verify task prefix was added
        call_args = mock_model.encode.call_args
        called_text = call_args[0][0]
        self.assertTrue(called_text.startswith("Represent this code for retrieving similar code: "))
        self.assertIn(text, called_text)
        
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.compile', side_effect=lambda x: x)
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_batch(self, mock_sentence_transformer_class, mock_compile, mock_cuda):
        """Test batch embedding generation."""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer_class.return_value = mock_model
        
        # Mock encode to return numpy array for batch
        # Return 2 embeddings of 896 dimensions each
        full_embeddings = np.array([[0.1] * 896, [0.2] * 896])
        mock_model.encode.return_value = full_embeddings
        
        # Initialize embedder
        model_id = "jinaai/jina-code-embeddings-0.5b"
        embedder = JinaCodeEmbedder(model_id=model_id, dimension=512)
        
        # Test batch embedding
        texts = ["Code snippet 1", "Code snippet 2"]
        embeddings = embedder.embed_batch(texts)
        
        # Verify results
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 512)
        self.assertEqual(len(embeddings[1]), 512)
        self.assertTrue(mock_model.encode.called)
        
        # Verify task prefix was added to all texts
        call_args = mock_model.encode.call_args
        called_texts = call_args[0][0]
        self.assertEqual(len(called_texts), 2)
        for called_text in called_texts:
            self.assertTrue(called_text.startswith("Represent this code for retrieving similar code: "))
        
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.compile', side_effect=lambda x: x)
    @patch('sentence_transformers.SentenceTransformer')
    def test_dimension(self, mock_sentence_transformer_class, mock_compile, mock_cuda):
        """Test dimension property."""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer_class.return_value = mock_model
        
        # Initialize embedder with custom dimension
        model_id = "jinaai/jina-code-embeddings-0.5b"
        embedder = JinaCodeEmbedder(model_id=model_id, dimension=256)
        
        # Verify dimension
        self.assertEqual(embedder.dimension, 256)
        
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.compile', side_effect=lambda x: x)
    @patch('sentence_transformers.SentenceTransformer')
    def test_task_prefix_application(self, mock_sentence_transformer_class, mock_compile, mock_cuda):
        """Test that task prefix is correctly applied."""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer_class.return_value = mock_model
        
        # Mock encode
        full_embedding = np.array([0.1] * 896)
        mock_model.encode.return_value = full_embedding
        
        # Initialize embedder
        model_id = "jinaai/jina-code-embeddings-0.5b"
        embedder = JinaCodeEmbedder(model_id=model_id)
        
        # Test with specific text
        text = "def hello_world(): pass"
        embedder.embed(text)
        
        # Verify the exact prefix format
        call_args = mock_model.encode.call_args
        called_text = call_args[0][0]
        expected_prefix = "Represent this code for retrieving similar code: "
        self.assertEqual(called_text, expected_prefix + text)
        
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.compile', side_effect=lambda x: x)
    @patch('sentence_transformers.SentenceTransformer')
    def test_model_initialization(self, mock_sentence_transformer_class, mock_compile, mock_cuda):
        """Test that model is initialized with correct parameters."""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer_class.return_value = mock_model
        
        # Initialize embedder
        model_id = "jinaai/jina-code-embeddings-0.5b"
        embedder = JinaCodeEmbedder(model_id=model_id, dimension=512)
        
        # Verify SentenceTransformer was called with correct parameters
        mock_sentence_transformer_class.assert_called_once()
        call_kwargs = mock_sentence_transformer_class.call_args[1]
        self.assertTrue(call_kwargs.get('trust_remote_code'))
        self.assertEqual(call_kwargs.get('device'), 'cpu')  # CPU when GPU not available
        
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.compile', side_effect=lambda x: x)
    @patch('sentence_transformers.SentenceTransformer')
    def test_matryoshka_truncation(self, mock_sentence_transformer_class, mock_compile, mock_cuda):
        """Test that embeddings are truncated to specified dimension."""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer_class.return_value = mock_model
        
        # Mock encode to return full 896-dimensional embedding
        full_embedding = np.array(list(range(896)))
        mock_model.encode.return_value = full_embedding
        
        # Initialize embedder with 512 dimensions
        model_id = "jinaai/jina-code-embeddings-0.5b"
        embedder = JinaCodeEmbedder(model_id=model_id, dimension=512)
        
        # Get embedding
        embedding = embedder.embed("test code")
        
        # Verify truncation - should only have first 512 values
        self.assertEqual(len(embedding), 512)
        self.assertEqual(embedding[0], 0)
        self.assertEqual(embedding[511], 511)
        
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.compile', side_effect=lambda x: x)
    @patch('sentence_transformers.SentenceTransformer')
    def test_default_parameters(self, mock_sentence_transformer_class, mock_compile, mock_cuda):
        """Test default parameter values."""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer_class.return_value = mock_model
        
        # Initialize embedder with model_id and default dimension
        model_id = "jinaai/jina-code-embeddings-0.5b"
        embedder = JinaCodeEmbedder(model_id=model_id)
        
        # Verify defaults
        self.assertEqual(embedder.model_id, model_id)
        self.assertEqual(embedder.dimension, 512)  # Default dimension
        self.assertEqual(embedder.task, "code2code")
        self.assertEqual(embedder.task_prefix, "Represent this code for retrieving similar code: ")
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="NVIDIA GeForce RTX 3080")
    @patch('torch.compile', side_effect=lambda x: x)
    @patch('sentence_transformers.SentenceTransformer')
    def test_gpu_detection(self, mock_sentence_transformer_class, mock_compile, mock_device_name, mock_cuda):
        """Test that GPU is used when available."""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer_class.return_value = mock_model
        
        # Initialize embedder
        model_id = "jinaai/jina-code-embeddings-0.5b"
        embedder = JinaCodeEmbedder(model_id=model_id)
        
        # Verify GPU device was selected
        self.assertEqual(embedder.device, "cuda")
        
        # Verify SentenceTransformer was initialized with cuda
        call_kwargs = mock_sentence_transformer_class.call_args[1]
        self.assertEqual(call_kwargs.get('device'), 'cuda')


if __name__ == '__main__':
    unittest.main()