# Use Node image with npm
FROM node:18

# Create app directory
WORKDIR /app

# Install app dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Compile TypeScript
RUN npx tsc

# Run the app
CMD ["node", "dist/index.js"]
