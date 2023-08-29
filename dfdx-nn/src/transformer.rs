use crate::*;

use dfdx::{
    dtypes::Dtype,
    shapes::Dim,
    tensor::{HasErr, PutTape, SplitTape, WithEmptyTape},
    tensor_ops::{Device, TryAdd},
};

#[derive(Clone, Debug, Sequential)]
#[built(FeedForward)]
pub struct FeedForwardConfig<Model: Dim, F: Dim> {
    pub l1: LinearConfig<Model, F>,
    pub act1: ReLU,
    pub l2: LinearConfig<F, Model>,
}

/// A single transformer encoder block
///
/// Generics
/// - `Model`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NumHeads`: The number of heads in [MultiHeadAttention].
/// - `F`: The size of the hidden layer in the feedforward network.
///
/// **Pytorch equivalent**:
/// ```python
/// encoder = torch.nn.TransformerEncoderLayer(
///    Model, NumHeads, dim_feedforward=F, batch_first=True, dropout=0.0
/// )
/// ```
#[derive(Clone, Debug, Sequential)]
#[built(EncoderBlock)]
pub struct EncoderBlockConfig<Model: Dim, NumHeads: Dim, F: Dim> {
    pub self_attn: ResidualAdd<MultiHeadAttentionConfig<Model, NumHeads>>,
    pub norm1: LayerNorm1DConfig<Model>,
    pub ff: ResidualAdd<FeedForwardConfig<Model, F>>,
    pub norm2: LayerNorm1DConfig<Model>,
}

impl<Model: Dim, NumHeads: Dim, F: Dim> EncoderBlockConfig<Model, NumHeads, F> {
    pub fn new(model: Model, num_heads: NumHeads, f: F) -> Self {
        EncoderBlockConfig {
            self_attn: ResidualAdd(MultiHeadAttentionConfig::new(
                model, num_heads, model, model,
            )),
            norm1: LayerNorm1DConfig(model),
            ff: ResidualAdd(FeedForwardConfig {
                l1: LinearConfig::new(model, f),
                act1: ReLU,
                l2: LinearConfig::new(f, model),
            }),
            norm2: LayerNorm1DConfig(model),
        }
    }
}

/// A transformer decoder block. Different than the normal transformer block
/// as this self attention accepts an additional sequence from the encoder.
///
/// Generics
/// - `Model`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NumHeads`: The number of heads in [MultiHeadAttention].
/// - `F`: The size of the hidden layer in the feedforward network.
///
/// **Pytorch equivalent**:
/// ```python
/// decoder = torch.nn.TransformerDecoderLayer(
///    Model, NumHeads, dim_feedforward=F, batch_first=True, dropout=0.0
/// )
/// ```
#[derive(Clone, Debug, CustomModule)]
#[built(DecoderBlock)]
pub struct DecoderBlockConfig<Model: Dim, NumHeads: Dim, F: Dim> {
    #[module]
    pub self_attn: ResidualAdd<MultiHeadAttentionConfig<Model, NumHeads>>,
    #[module]
    pub norm1: LayerNorm1DConfig<Model>,
    #[module]
    pub mh_attn: MultiHeadAttentionConfig<Model, NumHeads>,
    #[module]
    pub norm2: LayerNorm1DConfig<Model>,
    #[module]
    pub ff: ResidualAdd<FeedForwardConfig<Model, F>>,
    #[module]
    pub norm3: LayerNorm1DConfig<Model>,
}

impl<Model: Dim, NumHeads: Dim, F: Dim> DecoderBlockConfig<Model, NumHeads, F> {
    pub fn new(model: Model, num_heads: NumHeads, f: F) -> Self {
        DecoderBlockConfig {
            self_attn: ResidualAdd(MultiHeadAttentionConfig::new(
                model, num_heads, model, model,
            )),
            norm1: LayerNorm1DConfig(model),
            mh_attn: MultiHeadAttentionConfig::new(model, num_heads, model, model),
            norm2: LayerNorm1DConfig(model),
            ff: ResidualAdd(FeedForwardConfig {
                l1: LinearConfig::new(model, f),
                act1: ReLU,
                l2: LinearConfig::new(f, model),
            }),
            norm3: LayerNorm1DConfig(model),
        }
    }
}

impl<M: Dim, H: Dim, F: Dim, E: Dtype, D: Device<E>, Tgt, Mem> Module<(Tgt, Mem)>
    for DecoderBlock<M, H, F, E, D>
where
    Tgt: WithEmptyTape + SplitTape + TryAdd<Tgt::NoTape, Output = Tgt> + HasErr<Err = D::Err>,
    Mem: Clone,
    ResidualAdd<MultiHeadAttention<M, H, M, M, E, D>>: Module<Tgt, Output = Tgt, Error = D::Err>,
    MultiHeadAttention<M, H, M, M, E, D>: Module<(Tgt, Mem, Mem), Output = Tgt, Error = D::Err>,
    LayerNorm1D<M, E, D>: Module<Tgt, Output = Tgt, Error = D::Err>,
    ResidualAdd<FeedForward<M, F, E, D>>: Module<Tgt, Output = Tgt, Error = D::Err>,
{
    type Output = Tgt;
    type Error = D::Err;

    fn try_forward(&self, (tgt, mem): (Tgt, Mem)) -> Result<Self::Output, D::Err> {
        let x = self.self_attn.try_forward(tgt)?;
        let x = self.norm1.try_forward(x)?;

        let (x, tape) = x.split_tape();
        let x_residual = x.clone();
        let x = self
            .mh_attn
            .try_forward((x.put_tape(tape), mem.clone(), mem))?;
        let x = x.try_add(x_residual)?;
        let x = self.norm2.try_forward(x)?;
        let x = self.ff.try_forward(x)?;
        self.norm3.try_forward(x)
    }
}

/// Transformer architecture as described in
/// [Attention is all you need](https://arxiv.org/abs/1706.03762).
///
/// This is comprised of a [EncoderBlockConfig] and a [DecoderBlockConfig].
///
/// Generics:
/// - `Model`: Size of the input features to the encoder/decoder.
/// - `NumHeads`: Number of heads for [MultiHeadAttention].
/// - `F`: Feedforward hidden dimension for both encoder/decoder
///
/// **Pytorch equivalent**:
/// ```python
/// torch.nn.Transformer(
///     d_model=Model,
///     nhead=NumHeads,
///     num_encoder_layers=cfg.encoder.len(),
///     num_decoder_layers=cfg.decoder.len(),
///     dim_feedforward=F,
///     batch_first=True,
/// )
/// ```
#[derive(Clone, Debug, CustomModule)]
#[built(Transformer)]
pub struct TransformerConfig<Model: Dim, NumHeads: Dim, F: Dim> {
    #[module]
    pub encoder: Vec<EncoderBlockConfig<Model, NumHeads, F>>,
    #[module]
    pub decoder: Vec<DecoderBlockConfig<Model, NumHeads, F>>,
}

impl<Model: Dim, NumHeads: Dim, F: Dim> TransformerConfig<Model, NumHeads, F> {
    pub fn new(
        model: Model,
        num_heads: NumHeads,
        f: F,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
    ) -> Self {
        let mut encoder = Vec::with_capacity(num_encoder_layers);
        for _ in 0..num_encoder_layers {
            encoder.push(EncoderBlockConfig::new(model, num_heads, f));
        }
        let mut decoder = Vec::with_capacity(num_decoder_layers);
        for _ in 0..num_decoder_layers {
            decoder.push(DecoderBlockConfig::new(model, num_heads, f));
        }
        Self { encoder, decoder }
    }
}

impl<M: Dim, H: Dim, F: Dim, E: Dtype, D: Device<E>, Src: SplitTape, Tgt: PutTape<Src::Tape>>
    Module<(Src, Tgt)> for Transformer<M, H, F, E, D>
where
    Vec<EncoderBlock<M, H, F, E, D>>: Module<Src, Output = Src, Error = D::Err>,
    DecoderBlock<M, H, F, E, D>: Module<
        (<Tgt as PutTape<Src::Tape>>::Output, Src::NoTape),
        Output = <Tgt as PutTape<Src::Tape>>::Output,
        Error = D::Err,
    >,
{
    type Output = <Tgt as PutTape<Src::Tape>>::Output;
    type Error = D::Err;

    fn try_forward(&self, (src, tgt): (Src, Tgt)) -> Result<Self::Output, D::Err> {
        let (mem, tape) = self.encoder.try_forward(src)?.split_tape();
        let mut tgt = tgt.put_tape(tape);
        for block in self.decoder.iter() {
            tgt = block.try_forward((tgt, mem.clone()))?;
        }
        Ok(tgt)
    }
}
