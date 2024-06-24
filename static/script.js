function sendMessage() {
    var userMessage = document.getElementById("user-message").value;
    // Send the user's message to the server for processing
    fetch('/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userMessage })
        })
        .then(response => response.json())
        .then(data => {
            // Display the response from the server in the chat box
            var chatBox = document.getElementById("chat-box");
            chatBox.innerHTML += '<p>You: ' + userMessage + '</p>';
            chatBox.innerHTML += '<p>Chatbot: ' + data.message + '</p>';
        })
        .catch(error => console.error('Error:', error));
}