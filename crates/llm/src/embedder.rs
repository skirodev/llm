use std::path::Path;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[async_trait]
impl Embedder for RemoteEmbedder {
    fn embed(&self, data: &str) -> anyhow::Result<Embedding> {
        self.embedder.embed(data)
    async fn embed(&self, data: &str) -> anyhow::Result<Embedding> {
        self.embedder.embed(data).await
    }

    fn tokenizer(&self) -> &Tokenizer {
        self.embedder.tokenizer()
    }
    async fn batch_embed(&self, sequence: Vec<&str>) -> anyhow::Result<Vec<Embedding>> {
        Ok(self
            .make_request(ServerRequest { sequence })
            .await?
            .data
            .into_iter()
            .map(|p| p.embedding)
            .collect())
    }
}