import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score

def build_model(num_users, num_items, embedding_size=50):
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_size, name='item_embedding')(item_input)

    user_vector = Flatten(name='user_vector')(user_embedding)
    item_vector = Flatten(name='item_vector')(item_embedding)

    dot_product = Dot(axes=1, name='dot_product')([user_vector, item_vector])

    dense_1 = Dense(128, activation='relu')(dot_product)
    dense_2 = Dense(64, activation='relu')(dense_1)
    output = Dense(1, activation='sigmoid')(dense_2)

    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, train_data, epochs=10, batch_size=64):
    history = model.fit(
        [train_data['user'], train_data['item']], 
        train_data['interaction'], 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.2
    )
    return history

def evaluate_model(model, test_data):
    predictions = model.predict([test_data['user'], test_data['item']])
    predictions = (predictions > 0.5).astype(int)
    
    precision = precision_score(test_data['interaction'], predictions)
    recall = recall_score(test_data['interaction'], predictions)
    
    return precision, recall
