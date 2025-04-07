from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls

converter = DocumentConverter()

# --------------------------------------------------------------
# Basic PDF extraction
# --------------------------------------------------------------

# result = converter.convert("https://arxiv.org/pdf/2408.09869")
#
# document = result.document
# markdown_output = document.export_to_markdown()
# json_output = document.export_to_dict()
#
# print(markdown_output)

# --------------------------------------------------------------
# Basic HTML extraction
# --------------------------------------------------------------
# converter = DocumentConverter()
# links = ["https://www.galileo.ai/blog/mastering-agents-langgraph-vs-autogen-vs-crew",
#          "https://www.linkedin.com/pulse/langgraph-detailed-technical-exploration-ai-workflow-jagadeesan-n9woc",
#          "https://towardsdatascience.com/from-basics-to-advanced-exploring-langgraph-e8c1cf4db787",
#          "https://medium.com/@hao.l/why-langgraph-stands-out-as-an-exceptional-agent-framework-44806d969cc6",
#          "https://pub.towardsai.net/revolutionizing-project-management-with-ai-agents-and-langgraph-ff90951930c1",
#          "https://github.com/langchain-ai/langgraph"]
# result = converter.convert(links)
result = converter.convert("https://pub.towardsai.net/revolutionizing-project-management-with-ai-agents-and-langgraph-ff90951930c1")
document = result.document
markdown_output = document.export_to_markdown()
with open('../evaluator/mds/towardsai.md', 'w', encoding='utf-8') as f:
    f.write(markdown_output)

# --------------------------------------------------------------
# Scrape multiple pages using the sitemap
# --------------------------------------------------------------

# sitemap_urls = get_sitemap_urls("https://ds4sd.github.io/docling/")
# conv_results_iter = converter.convert_all(sitemap_urls)
#
# docs = []
# for result in conv_results_iter:
#     if result.document:
#         document = result.document
#         docs.append(document)
