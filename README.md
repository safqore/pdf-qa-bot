# pdf-qa-bot / Deployment branch


to run this the expectation is that you will have docker-compose installed.

if you don't you can install it with 

`brew install docker-compose`

once thats install all you need to do is

`docker-compose up --build`

depending on your set up, you might get a couple of pop ups you will have to accept for it to install the relevant packages

once this is up and running you will have these three docker containers

```
pdf-qa-bot-qdrant-1 
pdf-qa-bot-backend-1
pdf-qa-bot-frontend-1
```

once these are up and running the front end will auto open in your browser, incase it doesn't it can be accessed at 

http://localhost:3000/

this is a plain react / node website with a bot widget / script sending what ever is written in the text box to 

http://localhost:8000/query

This is a POST and the response back is whatever was sent to it


