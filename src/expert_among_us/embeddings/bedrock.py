import json
from botocore.exceptions import ClientError
from .base import Embedder
from typing import List, Optional, Callable
from ..utils.debug import DebugLogger
from ..utils.aws import AWSSessionManager, AWSCredentialError


class EmbedderError(Exception):
    """Base exception for embedder errors."""
    pass

class BedrockEmbedder(Embedder):
    def __init__(self, model_id: str, region_name: str = "us-west-2"):
        """Initialize Bedrock embedder with credential validation and auto-refresh.
        
        Args:
            model_id: Bedrock model ID for embeddings (e.g., amazon.titan-embed-text-v2:0)
            region_name: AWS region (default: us-west-2)
            
        Raises:
            EmbedderError: If AWS credentials are not found or invalid
        """
        self.model_id = model_id
        
        try:
            self._aws = AWSSessionManager(region_name)
        except AWSCredentialError as e:
            raise EmbedderError(str(e))
    
    @property
    def client(self):
        """Get Bedrock client with fresh credentials.
        
        This property ensures credentials are automatically refreshed from
        ~/.aws/credentials when they expire, enabling long-running processes
        to pick up externally updated credentials.
        """
        return self._aws.get_client('bedrock-runtime')
        
    def embed(self, text: str) -> List[float]:
        # Format request body for Amazon Titan Embed Text v2
        request_body = {
            "inputText": text
        }
        
        request = {
            "modelId": self.model_id,
            "body": request_body
        }
        
        # Log request if debug enabled
        request_id = None
        if DebugLogger.is_enabled():
            request_id = DebugLogger.log_request("bedrock", request, category="embedding")
        
        # Invoke model with properly formatted JSON body
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body)
        )
        
        # Parse response body
        response_body = json.loads(response['body'].read())
        embedding = response_body['embedding']
        
        # Log response if debug enabled
        if DebugLogger.is_enabled():
            DebugLogger.log_response("bedrock", {
                "embedding": embedding,
                "response_metadata": {
                    "model_id": self.model_id,
                    "embedding_dimension": len(embedding)
                }
            }, request_id, category="embedding")
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[List[float]]:
        embeddings = []
        total = len(texts)
        for i, text in enumerate(texts):
            embeddings.append(self.embed(text))
            if progress_callback:
                progress_callback(i + 1, total)
        return embeddings
    
    @property
    def dimension(self) -> int:
        return 1024