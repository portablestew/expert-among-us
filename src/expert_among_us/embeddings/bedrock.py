import boto3
import json
from .base import Embedder
from typing import List, Optional, Callable
from ..utils.debug import DebugLogger

class BedrockEmbedder(Embedder):
    def __init__(self, model_id: str):
        self.client = boto3.client('bedrock-runtime')
        self.model_id = model_id
        
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