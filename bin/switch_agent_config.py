#!/usr/bin/env python3
"""
Utility to switch the agent's retrieval configuration.
Usage: python bin/switch_agent_config.py [config_name]
"""

import yaml
import sys
import argparse
from pathlib import Path


def list_available_configs():
    """List all available retrieval configurations."""
    config_dir = Path("pipelines/configs/retrieval")
    if not config_dir.exists():
        print("❌ Retrieval configs directory not found")
        return []
    
    configs = []
    for config_file in config_dir.glob("*.yml"):
        config_name = config_file.stem
        
        # Read config to get description
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            retrieval_config = config.get('retrieval_pipeline', {})
            retriever_type = retrieval_config.get('retriever', {}).get('type', 'unknown')
            num_stages = len(retrieval_config.get('stages', []))
            
            description = f"{retriever_type} retrieval with {num_stages} stages"
            configs.append((config_name, description, str(config_file)))
            
        except Exception as e:
            configs.append((config_name, f"Error reading config: {e}", str(config_file)))
    
    return configs


def switch_agent_config(config_name: str):
    """Switch the agent's retrieval configuration."""
    # Check if config exists
    config_path = Path(f"pipelines/configs/retrieval/{config_name}.yml")
    if not config_path.exists():
        print(f"❌ Configuration '{config_name}' not found at {config_path}")
        return False
    
    # Load main config
    main_config_path = Path("config.yml")
    if not main_config_path.exists():
        print(f"❌ Main config file not found: {main_config_path}")
        return False
    
    try:
        with open(main_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update retrieval config path
        if 'retrieval' not in config:
            config['retrieval'] = {}
        
        old_config = config['retrieval'].get('config_path', 'not set')
        config['retrieval']['config_path'] = f"pipelines/configs/retrieval/{config_name}.yml"
        
        # Write updated config
        with open(main_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Agent retrieval configuration switched:")
        print(f"   From: {old_config}")
        print(f"   To:   pipelines/configs/retrieval/{config_name}.yml")
        
        # Show config details
        with open(config_path, 'r') as f:
            retrieval_config = yaml.safe_load(f)
        
        pipeline_config = retrieval_config.get('retrieval_pipeline', {})
        retriever_info = pipeline_config.get('retriever', {})
        stages = pipeline_config.get('stages', [])
        
        print(f"\n📋 Configuration Details:")
        print(f"   Retriever: {retriever_info.get('type', 'unknown')} (top_k={retriever_info.get('top_k', 5)})")
        print(f"   Stages: {len(stages)}")
        for i, stage in enumerate(stages, 1):
            stage_type = stage.get('type', 'unknown')
            print(f"     {i}. {stage_type}")
        
        print(f"\n🔄 Restart your agent to apply the new configuration.")
        return True
        
    except Exception as e:
        print(f"❌ Error updating configuration: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Switch agent retrieval configuration")
    parser.add_argument("config_name", nargs='?', help="Name of the configuration to switch to")
    parser.add_argument("--list", "-l", action="store_true", help="List available configurations")
    
    args = parser.parse_args()
    
    if args.list or not args.config_name:
        print("📋 Available Retrieval Configurations:")
        print("=" * 50)
        
        configs = list_available_configs()
        if not configs:
            print("❌ No configurations found")
            return
        
        for name, description, path in configs:
            print(f"🔧 {name}")
            print(f"   {description}")
            print(f"   Path: {path}")
            print()
        
        if not args.config_name:
            print("Usage: python bin/switch_agent_config.py <config_name>")
            return
    
    if args.config_name:
        success = switch_agent_config(args.config_name)
        if success:
            print(f"\n💡 Test the new configuration:")
            print(f"   python tests/test_agent_retrieval.py")
        else:
            print(f"\n📋 Available configs:")
            for name, _, _ in list_available_configs():
                print(f"   - {name}")


if __name__ == "__main__":
    main()
