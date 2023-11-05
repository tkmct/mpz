//! Oblivious Transfer for arithmetic values
use async_trait::async_trait;
use rand::{CryptoRng, Rng};

use mpz_circuits::arithmetic::{
    types::ArithValue,
    utils::{convert_value_to_crt, PRIMES},
};
use mpz_core::Block;
use mpz_garble_core::encoding::{
    add_label, cmul_label, crt_encoding_state as encoding_state, get_delta_by_modulus, CrtDelta,
    EncodedCrtValue, LabelModN,
};

/// Trait for send arithmetic value
#[async_trait]
pub trait OTSendCrtEncoding {
    /// Send list of encoded labels by OT
    /// 1. For each input, generate crt wires.
    /// 2. Represents the modulus value of each wire in bit representaiton.
    /// 3. Generates random zero label and one label for each bit wire.
    /// 4. Send the bit wires to receiver using OT.
    /// 5. Accumulate the zero labels in numeric form and returns it as vector.
    async fn send_arith<R: Rng + CryptoRng + ?Sized + Send>(
        &self,
        id: &str,
        input: Vec<EncodedCrtValue<encoding_state::Full>>,
        deltas: &[CrtDelta],
        rng: &mut R,
    ) -> Result<Vec<Vec<LabelModN>>, mpz_ot::OTError>;
}

#[async_trait]
impl<T> OTSendCrtEncoding for T
where
    T: mpz_ot::OTSenderShared<[Block; 2]> + Send + Sync,
{
    // for each encoded crt value in the vec
    // for each label in the encoded crt value
    // represent the modulus value in bit representation
    // create zero label and one label using the delta of the modulus
    // do the ot for each bit values
    async fn send_arith<R: Rng + CryptoRng + ?Sized + Send>(
        &self,
        id: &str,
        input: Vec<EncodedCrtValue<encoding_state::Full>>,
        deltas: &[CrtDelta],
        rng: &mut R,
    ) -> Result<Vec<Vec<LabelModN>>, mpz_ot::OTError> {
        let mut pairs = Vec::new();
        let mut bases = Vec::new();

        println!("calculate bit lables to send");
        for v in input.iter() {
            let mut base_v = Vec::new();
            for label in v.iter() {
                let q = label.modulus();
                let mut base = LabelModN::zero(q);

                let len = f32::from(q).log(2.0).ceil() as usize;

                for i in 0..len {
                    // get zero label and one label for each bit
                    let delta = get_delta_by_modulus(deltas, q).unwrap();
                    let zero = LabelModN::random(rng, q);
                    let one = add_label(&zero, &delta);

                    base = add_label(&base, &cmul_label(&zero, u64::pow(2, i as u32)));
                    pairs.push([zero.to_block(), one.to_block()]);
                }

                base_v.push(base);
            }
            bases.push(base_v);
        }

        println!("sending pairs");
        self.send(id, &pairs).await?;
        Ok(bases)
    }
}

/// A trait for receiving encodings via oblivious transfer.
#[async_trait]
pub trait OTReceiveCrtEncoding {
    /// Receives encodings from the sender.
    async fn receive_arith(
        &self,
        id: &str,
        choice: Vec<ArithValue>,
    ) -> Result<Vec<EncodedCrtValue<encoding_state::Active>>, mpz_ot::OTError>;
}

#[async_trait]
impl<T> OTReceiveCrtEncoding for T
where
    T: mpz_ot::OTReceiverShared<bool, Block> + Send + Sync,
{
    /// Receives encodings from the sender.
    async fn receive_arith(
        &self,
        id: &str,
        choice: Vec<ArithValue>,
    ) -> Result<Vec<EncodedCrtValue<encoding_state::Active>>, mpz_ot::OTError> {
        let mut lens = Vec::new();
        let mut bs = Vec::new();

        println!("deciding bit choices");
        choice.iter().for_each(|value| {
            convert_value_to_crt(*value)
                .iter()
                .enumerate()
                .for_each(|(idx, v)| {
                    // get prime f
                    let q = PRIMES[idx];
                    let len = f32::from(q).log(2.0).ceil() as usize;

                    for b in (0..len).map(|i| v & (1 << i) != 0) {
                        bs.push(b);
                    }
                    lens.push(len);
                });
        });

        println!("receiving the pairs");
        let mut blocks = self.receive(id, &bs).await?;
        println!("calculating back to the original value");

        // calculate each labels back to original crt label
        let mut labels = lens
            .iter()
            .enumerate()
            .map(|(i, &len)| {
                let q = PRIMES[i];
                let bits = blocks
                    .drain(0..len)
                    .map(|block| LabelModN::from_block(block, q))
                    .collect::<Vec<_>>();
                bits.iter()
                    .enumerate()
                    .map(|(j, bit_label)| cmul_label(bit_label, u64::pow(2, j as u32)))
                    .reduce(|acc, l| add_label(&acc, &l))
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let result = choice
            .iter()
            .map(|v| {
                EncodedCrtValue::<encoding_state::Active>::from(
                    labels.drain(..v.num_wire()).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use mpz_circuits::arithmetic::{types::ArithValue, utils::convert_value_to_crt};
    use mpz_garble_core::encoding::ChaChaCrtEncoder;
    use mpz_ot::mock::mock_ot_shared_pair;

    #[tokio::test]
    async fn test_bool_crt_transfer() {
        let encoder = ChaChaCrtEncoder::<1>::new([0; 32]);
        let mut rng = encoder.get_rng(0);
        let (sender, receiver) = mock_ot_shared_pair();

        let ty = vec![encoder.encode_by_len(0, 1)];

        let choices = vec![ArithValue::Bool(true)];

        // This bases should be used as actual low label for arith crt labels.
        let bases = sender
            .send_arith("", ty.clone(), encoder.deltas(), &mut rng)
            .await
            .unwrap();
        let inputs = bases
            .into_iter()
            .map(EncodedCrtValue::<encoding_state::Full>::from)
            .collect::<Vec<_>>();

        let received = receiver.receive_arith("", choices.clone()).await.unwrap();

        let expected = choices
            .into_iter()
            .zip(inputs)
            .map(|(choice, full)| full.select(encoder.deltas(), convert_value_to_crt(choice)))
            .collect::<Vec<_>>();

        assert_eq!(received, expected);
    }

    #[tokio::test]
    async fn test_crt_transfer() {
        let encoder = ChaChaCrtEncoder::<10>::new([0; 32]);
        let mut rng = encoder.get_rng(0);
        let (sender, receiver) = mock_ot_shared_pair();

        let ty = vec![encoder.encode_by_len(0, 10)];

        let choices = vec![ArithValue::U32(12)];

        // This bases should be used as actual low label for arith crt labels.
        let bases = sender
            .send_arith("", ty.clone(), encoder.deltas(), &mut rng)
            .await
            .unwrap();
        let inputs = bases
            .into_iter()
            .map(EncodedCrtValue::<encoding_state::Full>::from)
            .collect::<Vec<_>>();

        let received = receiver.receive_arith("", choices.clone()).await.unwrap();

        let expected = choices
            .into_iter()
            .zip(inputs)
            .map(|(choice, full)| full.select(encoder.deltas(), convert_value_to_crt(choice)))
            .collect::<Vec<_>>();

        assert_eq!(received, expected);
    }
}
