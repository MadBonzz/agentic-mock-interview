from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from dotenv import load_dotenv
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.utilities import StackExchangeAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

print(wikipedia.run("When would you choose a GAN over a Variational AutoEncoder?"))