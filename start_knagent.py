#!/usr/bin/env python3
"""
KnDAgent startup script
"""

import os
import sys
from knagent import KnDAgent

def main():
    """Main startup function"""
    print("🏥 KnDAgent - Knowledge Graph Agent")
    print("=" * 60)
    
    # Check necessary files
    doc_path = "./data/KnDAgent.docx"
    
    if not os.path.exists(doc_path):
        print(f"❌ Error: Document path does not exist: {doc_path}")
        print("Please ensure medical knowledge document is placed in the correct location")
        return
    
    print("✅ File check completed, starting Agent...")
    print()
    
    try:
        # Create and start agent
        agent = KnDAgent(doc_path=doc_path)
        
        # Start interactive mode
        agent.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted, exiting...")
    except Exception as e:
        print(f"\n❌ Agent startup failed: {e}")
        print("Please check error information and retry")

if __name__ == "__main__":
    main()
