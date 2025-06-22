import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- Define the CVAE model (same as in training) ----
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20 + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, y):
        h = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h = F.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# ---- Load model ----
device = torch.device("cpu")  # Use CPU in Streamlit
model = CVAE().to(device)
model.load_state_dict(torch.load("cvae_model.pt", map_location=device))
model.eval()

# ---- Streamlit UI ----
st.title("🖋️ Handwritten Digit Generator")
st.write("Select a digit (0–9) to generate 5 handwritten variations using a trained CVAE model.")

digit = st.selectbox("Choose a digit:", list(range(10)))

if st.button("Generate Images"):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        z = torch.randn(1, 20)
        y = torch.zeros(1, 10)
        y[0][digit] = 1
        with torch.no_grad():
            gen = model.decode(z, y).view(28, 28)
        axs[i].imshow(gen.cpu(), cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
