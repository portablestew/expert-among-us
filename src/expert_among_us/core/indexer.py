from typing import List
from expert_among_us.models.changelist import Changelist
from expert_among_us.embeddings.bedrock import BedrockEmbedder
from expert_among_us.vcs.git import GitVCS
from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB
from expert_among_us.db.vector.chroma import ChromaVectorDB
from expert_among_us.utils.progress import create_progress_bar, update_progress


class Indexer:
    def __init__(self, expert_config: dict):
        self.expert_config = expert_config
        self.vcs = GitVCS(expert_config['workspace_path'])
        self.metadata_db = SQLiteMetadataDB(expert_config['name'])
        self.vector_db = ChromaVectorDB(expert_config['name'])
        self.embedder = BedrockEmbedder(expert_config['embeddings_model_id'])
        
    def index(self):
        # Get commits from VCS
        changelists = self.vcs.get_commits(
            subdirs=self.expert_config['subdirs'],
            max_commits=self.expert_config['max_commits']
        )
        
        # Create progress bar
        progress, task_id = create_progress_bar("Indexing commits", len(changelists))
        progress.start()
        
        try:
            for changelist in changelists:
                # Generate metadata embedding
                metadata_text = changelist.get_metadata_text()
                metadata_embedding = self.embedder.embed(metadata_text)
                
                # Generate diff embedding if enabled
                if self.expert_config.get('embed_diffs', True):
                    diff_embedding = self.embedder.embed(changelist.diff)
                else:
                    diff_embedding = None
                
                # Store in databases
                self.metadata_db.insert_changelist(changelist, metadata_embedding)
                if diff_embedding:
                    self.vector_db.insert_vectors([changelist.id], [diff_embedding])
                
                # Update progress
                update_progress(progress, task_id, advance=1, description=f"Processing {changelist.id}")
        finally:
            progress.stop()