## **The Positional Encoding Layer in Transformers**


The Positional Encoding layer in Transformers plays a critical role by providing necessary positional information to the model. This is particularly important because the Transformer architecture, unlike RNNs or LSTMs, processes input sequences in parallel and lacks inherent mechanisms to account for the sequential order of tokens.

The mathematical intuition behind the Positional Encoding layer in Transformers is centered on enabling the model to incorporate information about the order of tokens in a sequence.


*  **Function Parameters**:
    
    position: Total positions or length of the sequence.
    d_model: Dimensionality of the model's output.

*  **Generating the Base Matrix**:

    angle_rads: Creates a matrix where rows represent sequence positions and columns represent feature dimensions. Values are scaled by dividing each position index by _10000 raised to (2 * index / d_model)._

*   **Applying Sine and Cosine Functions** :
    Even indices: Apply the sine function to encode positions.
    Odd indices: Apply the cosine function for a phase-shifted encoding.
        
    PE(pos, 2i) = sin(pos/1000<sup>(2i/dmodel)</sup> )

    PE(pos, 2i+1) = cos(pos/1000<sup>(2i/dmodel)</sup>)
    
*   **Creating the Positional Encoding Tensor**:
    The matrix is expanded to match input shape expectations of models like Transformers and cast to **np.float16**
        
*   **Output**:
    Returns a nd-array ready to be added to input embeddings to incorporate positional information.
    
    
    
**_NOTE_**: Please calculate the encodings using the above steps and dont just reshape and return. Also return a np.float16 array. 
Take the input embeddings, calculate the sin and cos and then return the encoding.
