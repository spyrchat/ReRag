from processors.dispatcher import ProcessorDispatcher

dispatcher = ProcessorDispatcher(chunk_size=300, chunk_overlap=30)
chunks = dispatcher.process_directory("sandbox")

print(f"Total Chunks: {len(chunks)}")
print(chunks[0].page_content)
