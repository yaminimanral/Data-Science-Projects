import discord
from discord.ext import commands
import logging
import asyncio
from query_engine.engine import query_knowledge
from preprocessing.extractor import extract_knowledge
from memory.mem0 import store_memory

logger = logging.getLogger(__name__)

class MemoryCommands(commands.Cog):
    """Commands for interacting with the Team Memory Bot."""
    
    def __init__(self, bot):
        self.bot = bot
    
    @commands.command(name="ask", help="Ask a question about team knowledge")
    async def ask(self, ctx, *, question):
        """
        Query the team knowledge memory.
        
        Args:
            question (str): Natural language question to ask the memory system
        """
        async with ctx.typing():
            try:
                # Inform user that processing is happening
                processing_msg = await ctx.send("Processing your question... 🧠")
                
                # Get response from query engine, passing the channel ID
                response = await query_knowledge(question, channel_id=str(ctx.channel.id))
                
                # Create embed for nicer display
                embed = discord.Embed(
                    title="Memory Response",
                    description=response,
                    color=discord.Color.blue()
                )
                embed.set_footer(text="Based on team memory records")
                
                await processing_msg.delete()
                await ctx.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                await ctx.send(f"Sorry, I encountered an error: {str(e)}")
    
    @commands.command(name="save", help="Save important knowledge to team memory")
    async def save(self, ctx, *, content):
        """
        Manually save knowledge to team memory.
        
        Args:
            content (str): Content to save to memory
        """
        async with ctx.typing():
            try:
                # Extract knowledge summary
                knowledge_summary = await extract_knowledge(content)
                
                # Store in memory with a single default type
                memory_entry = {
                    "type": "memory",
                    "summary": knowledge_summary,
                    "timestamp": ctx.message.created_at.isoformat(),
                    "context": f"channel: {ctx.channel.name}",
                    "participants": [f"@{ctx.author.name}"],
                    "raw_content": content,
                    "message_id": str(ctx.message.id),
                    "channel_id": str(ctx.channel.id),
                    "author_id": str(ctx.author.id)
                }
                
                await store_memory(memory_entry)
                
                # Confirmation message
                embed = discord.Embed(
                    title="Memory Saved",
                    description=f"**Summary:** {knowledge_summary}",
                    color=discord.Color.green()
                )
                await ctx.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error saving memory: {e}", exc_info=True)
                await ctx.send(f"Sorry, I encountered an error: {str(e)}")
    
    @commands.command(name="memory_help", help="Show help information about the memory bot")
    async def memory_help(self, ctx):
        """Display help information about the memory bot."""
        embed = discord.Embed(
            title="Team Memory Bot Help",
            description="I'm your team's knowledge memory bot. I track important information from your conversations.",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="How I Work",
            value="I monitor conversations and can store important information. "
                  "You can also manually save memories or ask questions about stored knowledge.",
            inline=False
        )
        
        embed.add_field(
            name="Commands",
            value=(
                f"`{ctx.prefix}ask [question]` - Ask a question about team knowledge\n"
                f"`{ctx.prefix}save [content]` - Manually save knowledge\n"
                f"`{ctx.prefix}memory_help` - Show this help message"
            ),
            inline=False
        )
        
        embed.add_field(
            name="Example Questions",
            value=(
                "• What did we discuss yesterday?\n"
                "• When did we decide to change the deadline?\n"
                "• What was our conclusion about the user interface?\n"
                "• What's the status of the project?"
            ),
            inline=False
        )
        
        await ctx.send(embed=embed)

async def setup(bot):
    """Add the commands cog to the bot."""
    await bot.add_cog(MemoryCommands(bot))
    logger.info("Memory commands registered")