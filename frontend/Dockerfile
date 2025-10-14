# Use the official Nginx image from Docker Hub
FROM nginx:alpine

# Remove the default nginx configuration file
RUN rm /etc/nginx/conf.d/default.conf

# Copy the custom nginx configuration file
COPY nginx.conf /etc/nginx/conf.d/

# Copy the static assets of the frontend application
COPY . /usr/share/nginx/html

# Expose port 80
EXPOSE 80 