# Use Node image with npm
FROM node:18

# Create app directory
WORKDIR /app

# Install app dependencies
COPY package*.json .
RUN npm install

# Copy source code
COPY . .

# Compile TypeScript
RUN npm run build
# RUN npx tsc

# Run the app
# RUN npm start
CMD ["npm", "run", "start"]
# CMD ["node", "dist/index.js"]
